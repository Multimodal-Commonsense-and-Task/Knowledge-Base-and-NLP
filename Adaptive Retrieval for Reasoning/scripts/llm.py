import re
import os
import time
import csv
import json
import torch
import random
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
# from ftfy import fix_text
from copy import deepcopy
from collections import defaultdict
from typing import Union, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer


class Base_LLM:
    def __init__(self, model_name_or_path, temperature=0.0, top_p=1.0, top_k=1, max_model_len=32768, 
                max_tokens=64, repetition_penalty=1.0, seed=42, **kwargs):
        self.kwargs = kwargs
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.repetition_penalty = repetition_penalty
        self.model_name_or_path = model_name_or_path
        self.vllm = self.kwargs.get("vllm", False)
        self.snowflake = self.kwargs.get("snowflake", False)
        self.peft_model_name = self.kwargs.get("peft_model_name", False)
        self.gpu_memory_utilization = self.kwargs.get("gpu_memory_utilization", 0.95)


        if self.snowflake:
            from snowflake.snowpark import Session
            from snowflake.cortex import complete
            self.complete = complete
            connection_params = {
                "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
                "user": os.environ.get("SNOWFLAKE_USERNAME"),
                "password": os.environ.get("SNOWFLAKE_PASSWORD"),
                "role": os.environ.get("SNOWFLAKE_ROLE"),
                "database": os.environ.get("SNOWFLAKE_DATABASE"),
                "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
                "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE")
            }
            snowpark_session = Session.builder.configs(connection_params).create()
            tokenizer_path = '/data/models/Llama-3.3-70B-Instruct'
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        elif self.vllm:
            self.lora_request = None
            if self.peft_model_name is not None:
                self.lora_request = LoRARequest("lora_adaptor", 1, self.peft_model_name)
            self.model = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                enable_lora=True if self.peft_model_name is not None else False, 
            )

        else:
            import deepspeed
            from torch.nn import DataParallel
            from peft import PeftModel, PeftConfig
            self.init_distributed()
            self.device = torch.cuda.current_device()

            config = PeftConfig.from_pretrained(self.peft_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype="bfloat16")
            self.model = PeftModel.from_pretrained(base_model, self.peft_model_name)
            self.model = self.model.merge_and_unload()
            
            self.model = deepspeed.init_inference(
                model=self.model,
                max_tokens=self.max_model_len, 
                mp_size=dist.get_world_size(),
                dtype="bfloat16",
                replace_method="auto",
                replace_with_kernel_inject=True
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
            self.tokenizer.padding_side = 'left'
            print(f"#> Load {self.model_name_or_path} / {self.peft_model_name} done")


    def llm_batch_inference(self, batch_prompts, stop=None, max_tokens=None, logprobs=None, return_n_tokens=False) -> List[str]:
        if self.snowflake:
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(self.run_single_prompt, i, prompt)
                    for i, prompt in enumerate(batch_prompts)
                ]
                results = [None] * len(batch_prompts)
                for future in as_completed(futures):
                    idx, response = future.result()
                    if stop is not None:
                        for stop_token in stop:
                            stop_token_idx = response.find(stop_token)
                            if stop_token_idx != -1:
                                response = response[:stop_token_idx]
                    results[idx] = response
                results = [{"text": results[i], "n_tokens": 0, "logprobs": None} for i in range(len(results))]
                return results

        elif self.vllm:
            if max_tokens is None:
                max_tokens = self.max_tokens
            output_list = self.model.generate(
                batch_prompts, 
                lora_request=self.lora_request, 
                sampling_params=SamplingParams(
                    max_tokens=max_tokens, 
                    logprobs=logprobs, 
                    temperature=self.temperature, 
                    top_p=self.top_p, 
                    top_k=self.top_k, 
                    repetition_penalty=self.repetition_penalty,
                    seed=self.seed,
                )
            )
            output_texts = [output.outputs[0].text for output in output_list]
            completion_tokens = [0 for _ in output_list]
            output_logprobs = [None for _ in output_list]
            if return_n_tokens:
                completion_tokens = [len(output.outputs[0].token_ids) for output in output_list]
            if logprobs is not None:
                output_logprobs = [output.outputs[0].logprobs for output in output_list]
            
            outputs = [{
                "text": text, 
                "n_tokens": n_tokens, 
                "logprobs": logprobs, 
            } for text, n_tokens, logprobs in zip(output_texts, completion_tokens, output_logprobs)]
            return outputs
        
        else:
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            
    def run_single_prompt(self, index, input_prompt):
        retries = 3
        while retries > 0:
            try:
                response = self.complete(
                    model=self.model_name_or_path,
                    prompt=input_prompt,
                    options={
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                )
                return index, response
            except Exception as e:
                print(e)
                time.sleep(0.1)
                
                # truncate
                full_prompt = self.tokenizer.apply_chat_template(
                    input_prompt, tokenize=False, add_generation_prompt=True,
                )
                if len(self.tokenizer.tokenize(full_prompt)) > 8000:
                    len_system_prompt = len(self.tokenizer.tokenize(input_prompt[0]['content']))
                    input_prompt[1]['content'] = self.truncate(input_prompt[1]['content'], 8000 - len_system_prompt)

                retries -= 1
                if retries == 0:
                    print(f"#> Error sample length of user prompt: {len(input_prompt[1]['content'])}")
                    print(f"#> Error sample content: {input_prompt[1]['content'][:1024]}")
                    return index, input_prompt[0]['content']

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def init_distributed(self):
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    def is_main_process(self):
        return dist.get_rank() == 0