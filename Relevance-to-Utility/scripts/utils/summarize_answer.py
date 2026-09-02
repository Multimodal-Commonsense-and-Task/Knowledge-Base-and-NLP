import csv
import json
import random
import torch
import re
import os, time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the pre-trained model."
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.0, 
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.8, 
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        '--subset_start_idx',
        type=int,
        default=0,
        help="Number of examples to process. Defaults to all if not specified."
    )    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    
    parser.add_argument(
        '--max_num_seqs', 
        type=int, 
        default=None, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--input_file_name',
        type=str,
        required=True,
        help="Name of input_file_name."
    )

    
    return parser.parse_args()

def main():
    args = parse_args()
    
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    max_num_seqs = args.max_num_seqs
    input_file_name = args.input_file_name
    
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0

    data_path = input_file_name
    
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    
    if max_num_seqs is not None:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            max_model_len=max_tokens,
            max_num_seqs=max_num_seqs,
            # dtype="float32",
        )
    else:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            max_model_len=max_tokens,
            # dtype="float32",
        )
    
    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
        ## filtered_data = sorted(filtered_data, key=lambda x: len(x["search_results"]), reverse=True)
        if subset_num != -1:
            filtered_data = filtered_data[args.subset_start_idx:args.subset_start_idx+subset_num]    
    print(f"Total number of examples: {len(filtered_data)}")
    # prepare input
    input_list = []
    for item in tqdm(filtered_data):
        question = item['Question']
        Output = item['Output']
        user_prompt = (f"Summarize the answer by returning only the final answer with no additional explanation or reasoning.(e.g. yes, no, noun phrase, etc.)\n"
                       f"Question: {question}\nanswer: {Output}\n\n"
                       "Final answer:")
        prompt = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    # Set default max_tokens if not provided
    # max_tokens = 64
    
    t_start = time.time()
    # Generate model outputs
    output_list = llm.generate(
        input_list, 
        sampling_params=SamplingParams(
            max_tokens=32,
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            repetition_penalty=repetition_penalty,
        ),
    )
    output_list = [output.outputs[0].text for output in output_list]

    total_time = time.time() - t_start
    
    for i in range(len(filtered_data)):
        filtered_data[i]['Original_Output'] = filtered_data[i]['Output']
        filtered_data[i]['Output'] = output_list[i]

    output_path = data_path.replace('.json', f'_summed.json')
    with open(output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
