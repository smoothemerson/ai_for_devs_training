# %%

import os

from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, DataCollatorWithPadding

# %%

notebook_login()

# %%
dataset = load_dataset(
    "json", data_files={"train": "./train.jsonl", "test": "./validation.jsonl"}
)
print(dataset)

# %%
