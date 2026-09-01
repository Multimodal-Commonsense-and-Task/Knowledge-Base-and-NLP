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
from evaluate import run_evaluation
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_haystack_rag_instruction,
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth', 'asqa', 'manualtc'],
        help="Name of the dataset to use."
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )
    
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
        '--ouptut_base_dir',
        type=str,
        default="runs.baselines",
        help="Name of the cached search results name."
    )

    parser.add_argument(
        '--input_file_name',
        type=str,
        required=True,
        help="Name of input_file_name."
    )

    parser.add_argument(
        '--num_sequences',
        type=int,
        default=1,
        help="num_sequences."
    )
    parser.add_argument(
        '--needle_in_haystack_test',
        action='store_true',
        help="Use needle_in_haystack_test."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    max_num_seqs = args.max_num_seqs
    ouptut_base_dir = args.ouptut_base_dir
    input_file_name = args.input_file_name
    num_sequences = args.num_sequences
    
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    # Paths to datasets
    if dataset_name == 'math500':
        data_path = f'./data/MATH500/{split}.json'
    elif dataset_name == 'gpqa':
        data_path = f'./data/GPQA/{split}.json'
    elif dataset_name == 'aime':
        data_path = f'./data/AIME/{split}.json'
    elif dataset_name == 'amc':
        data_path = f'./data/AMC/{split}.json'
    elif dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth', 'asqa']:
        data_path = f'./data/QA_Datasets/{dataset_name}/{dataset_name}.json'

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.direct'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/{ouptut_base_dir}/{dataset_name}.{model_short_name}.direct_gen'
        if args.needle_in_haystack_test:
            output_dir = f'./outputs/{ouptut_base_dir}/{dataset_name}.NIAH.{model_short_name}.direct_gen'

    os.makedirs(output_dir, exist_ok=True)
    
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
            dtype="float32",
        )
    
    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
        ## filtered_data = sorted(filtered_data, key=lambda x: len(x["search_results"]), reverse=True)
        if subset_num != -1:
            filtered_data = filtered_data[args.subset_start_idx:args.subset_start_idx+subset_num]    
    
    # prepare input
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'asqa']:
            if args.needle_in_haystack_test:
                user_prompt = get_haystack_rag_instruction(question)
            elif 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
            
        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)


        else:
            user_prompt = ""  # Default to empty if dataset not matched
        prompt = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 32768
            else:
                max_tokens = 25600
        else:
            max_tokens = 3096
    
    t_start = time.time()
    # Generate model outputs
    output_list = llm.generate(
        input_list, 
        sampling_params=SamplingParams(
            max_tokens=4096, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            repetition_penalty=repetition_penalty,
            n=num_sequences,
        )
    )
    ###############################################################
    # if num_sequences == 1:
    #     output_list = [output.outputs[0].text for output in output_list]
    #     for i, d in enumerate(filtered_data):
    #         d["Prev_Output"] = output_list[i]
    #     for i, d in enumerate(filtered_data):
    #         d["Prev_Input"] = input_list[i]
        
    # else:
    #     for i, d in enumerate(filtered_data):
    #         d["Prev_Input"] = input_list[i]
    #     round_2_filtered_data = []
    #     for i, output in enumerate(output_list):
    #         for output_text in output.outputs:
    #             output_text = output_text.text
    #             d = deepcopy(filtered_data[i])
    #             d["Prev_Output"] = output_text
    #             round_2_filtered_data.append(d)
    #     filtered_data = round_2_filtered_data
    #     output_list = [output_text.text for output in output_list for output_text in output.outputs]




    # t_start = time.time()
    # # Generate model outputs
    # output_list = llm.generate(
    #     input_list2, 
    #     sampling_params=SamplingParams(
    #         max_tokens=2048, 
    #         temperature=temperature, 
    #         top_p=top_p, 
    #         top_k=top_k, 
    #         repetition_penalty=repetition_penalty,
    #     )
    # )

    total_time = time.time() - t_start
    # if "manualtc" in dataset_name:
    #     split = f"{args.subset_start_idx}_{args.subset_start_idx+args.subset_num}"
    # Run evaluation
    run_evaluation(
        filtered_data, 
        input_list, 
        # input_list2, 
        output_list, 
        dataset_name, 
        output_dir, 
        total_time, 
        split,
    )

if __name__ == "__main__":
    main()
