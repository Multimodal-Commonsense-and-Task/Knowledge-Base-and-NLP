import re
import os
import sys
import csv
import time
import json
import torch
import random
import logging
import numpy as np
import transformers

from tqdm import tqdm
from ftfy import fix_text
from copy import deepcopy
from types import SimpleNamespace
from collections import defaultdict
from typing import Union, Tuple, Dict, List
from datasets import Dataset, DatasetDict

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback

from utils.utils import json_dump, jsonl_load
from utils.prompts import get_rerank_system_prompt, get_rerank_user_prompt

logger = logging.getLogger(__name__)

class DebugCollator:
    def __init__(self, base_collator, tokenizer):
        self.base_collator = base_collator
        self.tokenizer = tokenizer

    def __call__(self, features):
        example = features[0]

        if "prompt" in example:
            print("==== Raw prompt ====")
            print(example["prompt"])
        if "completion" in example:
            print("==== Raw completion ====")
            print(example["completion"])

        batch = self.base_collator(features)

        print("==== Decoded input_ids ====")
        print(self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))

        print("==== Labels ====")
        print(self.tokenizer.decode([id if id != -100 else self.tokenizer.pad_token_id for id in batch['labels'][0]], skip_special_tokens=False))
        print("=" * 80)
        assert False

        return batch


class Trainer:
    def __init__(self, model_name_or_path, prompt_mode, batch_size, gradient_accumulation_steps, 
                ds_config, output_dir, train_data_path, num_train_epochs, max_qlen=1024, max_dlen=1024, 
                **kwargs):
        self.kwargs = kwargs
        self.max_qlen = max_qlen
        self.max_dlen = max_dlen
        self.ds_config = ds_config
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.prompt_mode = prompt_mode
        self.num_train_epochs = num_train_epochs
        self.model_name_or_path = model_name_or_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        # Dataset
        self.train_dataset = self.get_datasets(train_data_path)

        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="bfloat16",
        )

        lora_config = LoraConfig(
            base_model_name_or_path=self.model_name_or_path,
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
        lora_model = get_peft_model(model, lora_config)

        training_args = SFTConfig(
            output_dir=self.output_dir,
            bf16=True,
            packing=False,
            # save_steps=500,
            max_length=8192, 
            logging_steps=10, 
            ddp_timeout=18000, 
            learning_rate=1e-5, 
            dataset_num_proc=32, 
            deepspeed=self.ds_config,
            eos_token=self.tokenizer.eos_token,
            num_train_epochs=self.num_train_epochs, 
            per_device_train_batch_size=self.batch_size,
            gradient_checkpointing=False,  # True gives error https://github.com/huggingface/trl/issues/2819
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        instruction_template = "<|im_start|>system"
        response_template = "<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                                response_template=response_template,
                                                tokenizer=self.tokenizer)
        # collator = DebugCollator(collator, self.tokenizer)

        self.trainer = SFTTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=collator, 
        )

    def train(self):
        self.trainer.train()

    def get_datasets(self, train_data_path):
        raw_dataset = jsonl_load(train_data_path)
        # import pdb; pdb.set_trace()
        print(raw_dataset[0])
        dataset = Dataset.from_list(raw_dataset)
        dataset = dataset.map(self.build_prompt)
        dataset = dataset.remove_columns(['query', 'trace', 'analysis', 'docs', 'raw_docs'])
        # dataset = dataset.remove_columns(['query', 'trace', 'docs', 'raw_docs']) #  'analysis'
        return dataset
        
    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def build_prompt(self, data):
        query = data['query']
        trace = data['trace']
        docs = data['docs']
        analysis = data['analysis']
        raw_docs = data['raw_docs']

        # prompt
        messages = []
        messages.append({"role": "system", "content": get_rerank_system_prompt(self.prompt_mode, n_docs=len(raw_docs))})

        documents = ""
        for idx, doc in enumerate(raw_docs):
            if 'content' in doc:
                contents = doc['content'].strip()
            elif 'contents' in doc:
                contents = doc['contents'].strip()
            else:
                assert False
            contents = self.truncate(contents, self.max_dlen)
            documents += f"[{idx+1}]: {contents}\n"
        query = self.truncate(query, self.max_qlen)
        query = query.strip()
        documents = documents.strip()
        
        user_prompt = fix_text(get_rerank_user_prompt(
            query, 
            documents, 
            self.prompt_mode, 
        ))
        messages.append({"role": "user", "content": user_prompt})

        # completion
        completion = ""
        reasoning_steps = "[Reasoning Trace]\n"
        for idx, step in enumerate(trace):
            reasoning_steps += f"Step {idx+1}: {step}\n"
        reasoning_steps = reasoning_steps.strip()


        ranking = "[Document Ranking]\n"
        did2idx = {doc['id']:f"[{idx+1}]"for idx, doc in enumerate(raw_docs)}
        ranked_dids = [did2idx[doc['id']] for doc in docs]
        ranking += " > ".join(ranked_dids)
        ranking = ranking.strip()

        completion = reasoning_steps + '\n\n' + ranking
        completion = [{'role': 'assistant', 'content': completion}]
        
        example = {"prompt": messages, "completion": completion}
        return example
