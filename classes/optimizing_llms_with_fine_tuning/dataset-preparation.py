# %%
import json

from datasets import load_dataset

# %%
datasets = load_dataset(
    "hate-speech-portuguese/hate_speech_portuguese",
    revision="refs/convert/parquet",
    split="train[:10%]",
)

# %%
print(datasets)

# %%
datasets = datasets.remove_columns(
    [
        "hatespeech_G1",
        "annotator_G1",
        "hatespeech_G2",
        "annotator_G2",
        "hatespeech_G3",
        "annotator_G3",
    ]
)

# %%
datasets = datasets.train_test_split(test_size=0.2)


# %%
def removeN(example):
    example["text"] = example["text"].replace("\n", "")
    return example


# %%
datasets = datasets.map(removeN)


# %%
# label 0 -> No Hate Speech
# label 1 -> Hate Speech
def labelChange(example):
    example["label_text"] = "No Hate Speech" if example["label"] == 0 else "Hate Speech"
    return example


# %%
datasets = datasets.map(labelChange)

# %%
datasets = datasets.remove_columns(["label"])

# %%
print(datasets)


# %%
# Construção do objeto para openai
def dataset_to_jsonl(dataset, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for example in dataset:
            json_obj = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Seu trabalho é classificar os comentários dos usuários em Hate Speech ou No Hate Speech.",
                    },
                    {"role": "user", "content": example["text"]},
                    {"role": "assistant", "content": example["label_text"]},
                ]
            }

            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


# %%
dataset_to_jsonl(datasets["train"], "train.jsonl")

# %%
dataset_to_jsonl(datasets["test"], "validation.jsonl")


# %%
# Construção do objeto para AWS
def dataset_to_jsonl_aws(dataset, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for example in dataset:
            json_obj = {"prompt": example["text"], "completion": example["label_text"]}

            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


# %%
dataset_to_jsonl_aws(datasets["train"], "train.jsonl")

# %%
dataset_to_jsonl_aws(datasets["test"], "validation.jsonl")

# %%
