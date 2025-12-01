# %%

import os

import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset

# %%
dataset = load_dataset(
    "json",
    data_files={
        "train": "./datasets/train.jsonl",
        "test": "./datasets/test.jsonl",
    },
)
print(dataset)

# %%
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

mapDict = {"suporte": 0, "venda": 1}


def transform_labels(label):
    label = label["completion"]
    result = []

    for label_text in label:
        result.append(mapDict[label_text])

    return {"label": result}


def tokenize_function(example):
    return tokenizer(example["prompt"], padding=True, truncation=True)


# %%
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(transform_labels, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
output_dir = "./bert-hate-speech-test"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=2,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=3,
)

os.environ["WANDB_DISABLE"] = "true"
os.environ["WANDB_MODE"] = "offline"

# %%
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %%
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# %%
trainer.train()

# %%
trainer.evaluate()

# %%
trainer.save_model()

# %%
