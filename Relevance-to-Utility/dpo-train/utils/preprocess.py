import os
import re
import json

from tqdm import tqdm
from ftfy import fix_text
from typing import Literal
from datasets import Dataset, DatasetDict


def jsonl_load(path):
    data = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_raw_datasets(script_args, dataset_name, tokenizer):
    dataset = jsonl_load(os.path.join(script_args.dataset_dir, dataset_name))

    raw_dataset = []
    for data in tqdm(dataset):
        input_context = data['prompt']
        messages = [
            {"role": "user", "content": input_context}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)

        chosen = data['chosen'] + tokenizer.eos_token
        rejected = data['rejected'] + tokenizer.eos_token

        raw_dataset.append({'prompt':prompt, 'chosen':chosen, 'rejected':rejected})

    raw_datasets = DatasetDict({'train': Dataset.from_list(dataset)})
    return raw_datasets

