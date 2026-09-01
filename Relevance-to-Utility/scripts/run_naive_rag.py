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
import torch
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_naive_rag_instruction, 
    get_haystack_rag_instruction,
    get_pathrag_instruction, 
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
                 'msmarco', 'msmarco_abs_500', 'msmarco_ext_500', 'msmarco_train_abs_500', 'msmarco_abs_500_per_doc',
                 'crag', 'ambigqa'],
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
        '--context_cache_name',
        default=None,
        type=str,
        help="Name of the cached context for graphRAG"
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
    context_cache_name = args.context_cache_name

    print(f"args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Paths to datasets
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name.split("_")[0]}/{dataset_name}.json'

    # ---------------------- Caching Mechanism ----------------------
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
    
    if context_cache_name is not None:
        context_cache_name = os.path.join(cache_dir, dataset_name.split("_")[0], context_cache_name)
        with open(context_cache_name, 'r', encoding='utf-8') as f:
            context_cache = json.load(f)
    else:
        context_cache = {}


    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.naive_rag'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.naive_rag'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/{ouptut_base_dir}/{dataset_name.split("_")[0]}.{model_short_name}.naive_rag'
    output_dir += f"/{search_cache_name[:-5]}/"

    if top_k != 10:
        output_dir += f"top{top_k}/"
        
    print(f"#> output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Data Loading ----------------------
    print(f"#> data_path: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if subset_num != -1:
            data = data[args.subset_start_idx:args.subset_start_idx+subset_num]

    # ---------------------- Search and Document Retrieval ----------------------
    print("Performing Bing Web Searches for all questions...")

    # Initialize a list to hold relevant information for each question
    all_relevant_info = []

    for item in tqdm(data, desc="Searching"):
        qid = item['id']
        question = item['Question']
        # Check if the question has already been searched and cached
        
        if question in search_cache:
            results = search_cache[question]
            # print(f"Using cached search results for question: {question}")
        # if qid in search_cache:
        #     results = search_cache[qid]

        elif "per_doc" in input_file_name:
            assert "sub_id" in item, item
            qid, sub_qid = item["id"], item["sub_id"]
            qid = "-".join([str(qid), str(sub_qid)])
            results = search_cache[qid]
        
        else:
            assert False, f"The retrieved results should be prepared in advance. question: {question} / search_cache_path: {search_cache_path}"
            if dataset_name == 'livecode':
                search_question = question[:500]
            else:
                search_question = question
            results = bing_web_search(search_question, bing_subscription_key, bing_endpoint, market='en-US', language='en')
            search_cache[question] = results
            # print(f"Executed and cached search for question: {question}")

        # Extract relevant information from search results
        # relevant_info = extract_relevant_info(results)[:top_k]
        # longer result come first
        if "sort" in search_cache_name:
            results = sorted(results, key=lambda x: len(x['contents']), reverse=True)
        relevant_info = results[:top_k]
        all_relevant_info.append(relevant_info)


    del search_cache
    gc.collect()
    # ---------------------- Prompt Construction ----------------------
    print("Constructing prompts for generation...")
    if context_length < 1000:
        context_length = 1024 * context_length
    input_prompts = []

    for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
        qid = item['id']
        question = item['Question']

        formatted_documents = ""
        relevant_info = all_relevant_info[idx]

        for i, doc_info in enumerate(relevant_info):
            document = f"**Document {i + 1}:**\n"
            if len(doc_info.get('title', '')) > 0:
                document += f"**Title:** {doc_info.get('title', '')}\n"
            document += f"**Content:** {doc_info.get('contents', '')}\n\n"

            formatted_documents += document

        if context_length:
            formatted_documents = formatted_documents[:1024 * 128] # To speed up, truncate context with 128k character
            tokenized = tokenizer.encode(formatted_documents, add_special_tokens=False)
            truncated = tokenized[:context_length]
            formatted_documents = tokenizer.decode(truncated, skip_special_tokens=True)

        cached_context = context_cache.get(question, '')
        if len(cached_context) > 0 and cached_context != "Sorry, I'm not able to provide an answer to that question.":
            instruction = get_pathrag_instruction(question, cached_context)
            # instruction = get_haystack_rag_instruction(question, cached_context, dataset_name=dataset_name)
        else:
            instruction = get_haystack_rag_instruction(question, formatted_documents, dataset_name=dataset_name)
        user_prompt = ""  # Default to empty if dataset not matched


        # Combine instruction and user prompt
        full_prompt = instruction + "\n\n" + user_prompt

        # Apply tokenizer and chat template
        prompt = [{"role": "user", "content": full_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_prompts.append(prompt)


    # ---------------------- Generation ----------------------
    # Initialize the LLM
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

    print("Generating answers with LLM...")

    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower():
            max_tokens = 20480
        else:
            max_tokens = 10240

    start_time = time.time()
    # Generate model outputs
    output_list = llm.generate(
        input_prompts, 
        sampling_params=SamplingParams(
            max_tokens=1024, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k_sampling, 
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
    )
    lengths = [len(output.prompt_token_ids) for output in output_list]
    print(f"avg. lengths: {np.mean(lengths)}")

    total_time = time.time() - start_time

    # ---------------------- Evaluation ----------------------
    print("Evaluating generated answers...")
    run_evaluation(
        filtered_data=data,
        input_list=input_prompts,
        output_list=output_list,
        dataset_name=dataset_name,
        output_dir=output_dir,
        total_time=total_time,
        split=split,
    )

    print("Process completed.")

if __name__ == "__main__":
    main()
