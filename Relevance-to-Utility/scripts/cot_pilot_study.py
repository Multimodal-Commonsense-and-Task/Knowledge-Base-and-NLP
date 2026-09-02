# run_naive_rag.py
import os
import json
import time
import random
random.seed(42)
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse
from evaluate import run_evaluation, extract_answer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc
import re
import string
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import torch

from snowflake.snowpark import Session
from snowflake.cortex import complete
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_task_instruction_cot(document: str, question: str, mode: str) -> str:
    return(
        f"Think step by step to use the provided documents answer a user's question.\n\n"
        f"Question: {question}\n\n"
        f"Documents:\n{document}\n\n"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 
                 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth', 'asqa',
                 'manualtc', 'msmarco', 'msmarco_abs_500', 'msmarco_ext_500', 'crag', 'hotpotqatrainhard', 'msmarco_train_abs_full',
                 'crag_500', 'msmarco_train_ext_full', "mmlu_all_qa_500","mmlu_train_40000" ],
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
        '--subset_start_idx',
        type=int,
        default=0,
        help="Number of examples to process. Defaults to all if not specified."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Number of top search results to retrieve."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
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
        '--top_k_sampling',
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

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        required=False,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    parser.add_argument(
        '--search_cache_name',
        type=str,
        required=True,
        help="Name of the cached search results name."
    )

    parser.add_argument(
        '--max_num_seqs', 
        type=int, 
        default=None, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help="Random seed for generation."
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
        default="runs.baselines",
        help="Name of the cached search results name."
    )
    
    # for needle-in-the-haystack test
    parser.add_argument(
        '--context_length',
        default=0,
        type=int,
        help="number of tokens to fill the haystack with"
    )

    parser.add_argument(
        '--snowflake',
        action='store_true',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default = "",
        help="mode for the prompt"
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--window_overlap',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--mmlu_style',
        type=str,
        default = '',
        choices=['', 'qonly', 'qandc'],
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key

    max_num_seqs = args.max_num_seqs
    search_cache_name = args.search_cache_name
    seed = args.seed
    ouptut_base_dir = args.ouptut_base_dir
    input_file_name = args.input_file_name
    context_length = args.context_length
    subset_start_idx = args.subset_start_idx
    mode = args.mode

    print(f"args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    data_path = f'./data/QA_Datasets/{dataset_name.split("_")[0]}/{dataset_name}.json'


    # ---------------------- Data Loading ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, dataset_name.split("_")[0], search_cache_name)

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    print(f"#> search_cache_path: {search_cache_path}")
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.naive_rag'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.naive_rag'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/{ouptut_base_dir}/{dataset_name.split("_")[0]}.cot'
    output_dir += f"/{search_cache_name[:-5]}/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"#> data_path: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if subset_num != -1:
            data = data[args.subset_start_idx:args.subset_start_idx+subset_num]

    # ---------------------- Search and Document Retrieval ----------------------
    print("Performing Bing Web Searches for all questions...")

    all_relevant_info = []

    for item in tqdm(data, desc="Searching"):
        question = item['Question']
        
        if question in search_cache:
            results = search_cache[question]
        elif ("mmlu" in dataset_name and question.split("\n\n")[-1].strip() in search_cache):
            results = search_cache[question.split("\n\n")[-1].strip()]
        else:
            assert False, f"The retrieved results should be prepared in advance. question: {question} / search_cache_path: {search_cache_path}"
            if dataset_name == 'livecode':
                search_question = question[:500]
            else:
                search_question = question
            results = bing_web_search(search_question, bing_subscription_key, bing_endpoint, market='en-US', language='en')
            search_cache[question] = results

        relevant_info = results[:top_k]
        all_relevant_info.append(relevant_info)

    del search_cache
    gc.collect()


    # ---------------------- Model Loading ----------------------
    # Initialize the LLM
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


    # ---------------------- Prompt Construction ----------------------
    if context_length < 1000:
        context_length = 1024 * context_length

    input_prompts = []
    input_info = []

    for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
        qid = item['id']
        question = item['Question']
        relevant_info = all_relevant_info[idx]

        documents = []
        for i, doc_info in enumerate(relevant_info):
            docid = doc_info['id']
            doctitle = doc_info.get('title', '')
            document = doc_info['contents']
            documents.append(f"Document {i+1}: {document}")
        document = "\n\n".join(documents)

        full_prompt = get_task_instruction_cot(
            document=document, 
            question=question,
            mode=mode,
        )
        full_prompt = [{"role": "user", "content": full_prompt}]
        input_prompts.append(full_prompt)
        input_info.append([question, qid, docid, doctitle])

    print(f"Total number of prompts: {len(input_prompts)}")
    # ---------------------- Generation ----------------------
    print("Generation...")
    start_time = time.time()

    def run_single_prompt(index, input_prompt):
        retries = 3
        while retries > 0:
            try:
                response = complete(
                    model=model_path,
                    prompt=input_prompt,
                    options={
                        "max_tokens": context_length,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
                return index, response
            except Exception as e:
                print(e)
                retries -= 1
                time.sleep(0.05)  # wait before retrying
                if retries == 0:
                    return index, input_prompt[0]['content']

    NUM_THREADS = 5
    results = [None] * len(input_prompts)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(run_single_prompt, i, prompt) for i, prompt in enumerate(input_prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            index, result = future.result()
            results[index] = result

    output_list = results

    # ---------------------- Saving Output ----------------------
    for data_item, input_prompt, output in zip(data, input_prompts, output_list):
        data_item['input'] = input_prompt
        data_item['output'] = output
            
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    output_path = os.path.join(output_dir, f'{input_file_name}_{model_short_name}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saving {output_path} done.")

if __name__ == "__main__":
    main()
