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
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_naive_rag_instruction, 
    get_haystack_rag_instruction,
    get_task_instruction_naive_rewrite_docs,
)
from snowflake.snowpark import Session
from snowflake.cortex import complete
from concurrent.futures import ThreadPoolExecutor, as_completed


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
                 'crag_500', 'ambigqa', 'msmarco_train_ext_full', 'ambigqa'],
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

    # Paths to datasets
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
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

    if args.snowflake:
        pass
    else:
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
    if args.snowflake:
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
    else:
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


    # ---------------------- Prompt Construction ----------------------
    if context_length < 1000:
        context_length = 1024 * context_length

    if 'conditional' in mode or 'point' in mode or 'rewriter' in mode:
        input_prompts = []
        input_info = []

        for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
            qid = item['id']
            question = item['Question']
            relevant_info = all_relevant_info[idx]

            for i, doc_info in enumerate(relevant_info):
                docid = doc_info['id']
                doctitle = doc_info.get('title', '')
                document = doc_info['contents']

                previous_documents = ""
                previous_doc_info = relevant_info[:i] + relevant_info[i+1:]
                for j, doc_info in enumerate(previous_doc_info):
                    previous_documents += f"Document {j+1}:\n"
                    previous_documents += f"{doc_info['contents']}\n\n"

                full_prompt = get_task_instruction_naive_rewrite_docs(
                    document=document, 
                    question=question, 
                    other_documents=previous_documents, 
                    mode=mode
                )
                
                full_prompt = [{"role": "user", "content": full_prompt}]
                if not args.snowflake:
                    full_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=True)

                input_prompts.append(full_prompt)
                input_info.append([question, qid, docid, doctitle])

            
        print(f"Total number of prompts: {len(input_prompts)}")
        # ---------------------- Generation ----------------------
        print("Generation...")
        start_time = time.time()

        if args.snowflake:
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

            NUM_THREADS = 128
            results = [None] * len(input_prompts)

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [executor.submit(run_single_prompt, i, prompt) for i, prompt in enumerate(input_prompts)]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    index, result = future.result()
                    results[index] = result

            output_list = results
        else:
            output_list = llm.generate(
                input_prompts, 
                sampling_params=SamplingParams(
                    max_tokens=context_length, 
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k_sampling, 
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                )
            )
            lengths = [len(output.prompt_token_ids) for output in output_list]
            output_list = [output.outputs[0].text for output in output_list]
            print(f"avg. lengths: {np.mean(lengths)}")

        # ---------------------- Saving Output ----------------------
        rewritten_docs = []
        rewritten_cache = defaultdict(list)

        for (question, qid, docid, title), input_prompt, result in zip(input_info, input_prompts, output_list):
            rewritten_docs.append({'qid':qid, 'docid':docid, 'Input':input_prompt, 'rewritten_contents':result})
            rewritten_cache[question].append({'id':docid, 'title':title, 'contents':result})

    elif 'list' in mode:
        rewritten_docs = []
        rewritten_cache = defaultdict(list)

        for i in range(0, top_k, args.window_size - args.window_overlap):
            input_prompts = []
            input_info = []

            for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
                qid = item['id']
                question = item['Question']
                relevant_info = all_relevant_info[idx]

                if i + args.window_size > len(relevant_info):
                    if i + args.window_overlap >= len(relevant_info):
                        print(f"All the documents are already included in the previous window. Skipping question: {question}")
                        continue
                    elif args.window_overlap > 0:
                            doc_infos = relevant_info[len(relevant_info)-args.window_size:]
                    else:
                        doc_infos = relevant_info[i:i+args.window_size]
                else:
                    doc_infos = relevant_info[i:i+args.window_size]

                docid = doc_infos[0]['id']
                doctitle = doc_infos[0].get('title', '')
                document = ""
                for j, doc_info in enumerate(doc_infos):
                    if "cot" in mode:
                        document += f"Document {j+1}: {doc_info['contents']}\n\n"
                    else:
                        document += f"Target Document {j+1}: {doc_info['contents']}\n\n"
                other_doc_infos = [doc_info for doc_info in relevant_info if doc_info not in doc_infos]
                other_document = ""
                for j, doc_info in enumerate(other_doc_infos):
                    other_document += f"Other Document {j+1}: {doc_info['contents']}\n\n"

                full_prompt = get_task_instruction_naive_rewrite_docs(
                    document=document, 
                    question=question, 
                    other_documents=other_document, 
                    mode=mode,
                    num_of_docs=len(doc_infos), 
                    other_rewritten_docs= True if ("v3" in mode and i > 0) else False,
                )

                full_prompt = [{"role": "user", "content": full_prompt}]
                if not args.snowflake:
                    full_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=True)
                input_prompts.append(full_prompt)
                input_info.append([question, qid, docid, doctitle])

            # ---------------------- Generation ----------------------
            print("Generation...")
            start_time = time.time()

            if args.snowflake:
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
                            if retries == 0:
                                return index, input_prompt[0]['content']

                NUM_THREADS = 64
                results = [None] * len(input_prompts)

                with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                    futures = [executor.submit(run_single_prompt, i, prompt) for i, prompt in enumerate(input_prompts)]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                        index, result = future.result()
                        results[index] = result

                output_list = results
            else:
                # Generate model outputs
                output_list = llm.generate(
                    input_prompts, 
                    sampling_params=SamplingParams(
                        max_tokens=context_length, 
                        temperature=temperature, 
                        top_p=top_p, 
                        top_k=top_k_sampling, 
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                    )
                )
                lengths = [len(output.prompt_token_ids) for output in output_list]
                output_list = [output.outputs[0].text for output in output_list]
                print(f"avg. lengths: {np.mean(lengths)}")

            # ---------------------- Saving Output ----------------------
            for (question, qid, docid, title), input_prompt, result in zip(input_info, input_prompts, output_list):
                rewritten_docs.append({'qid':qid, 'docid':docid, 'Input':input_prompt, 'rewritten_contents':result})
                rewritten_cache[question].append({'id':docid, 'title':title, 'contents':result,})
        # input: rewritten docs & query. output: next query to be used for the next window
        if i + args.window_size - args.window_overlap < len(relevant_info): # if there are more documents to process
            input_prompts = []
            input_info = []


    else:
        assert False, 'args.mode should be choosen among [point, conditional, list, rewriter]'

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    search_cache_name = f'{search_cache_name[:-5]}_{mode}_rewritten_{model_short_name}_{subset_start_idx}.json'
    if 'list' in mode:
        search_cache_name = f'{search_cache_name[:-5]}_ws{args.window_size}_wo{args.window_overlap}.json'

    output_path = f'./data/QA_Datasets/{dataset_name.split("_")[0]}/{search_cache_name}'
    with open(output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(rewritten_docs, json_file, indent=4, ensure_ascii=False)

    output_path = f'./cache/{dataset_name.split("_")[0]}/{search_cache_name}'
    with open(output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(rewritten_cache, json_file, indent=4, ensure_ascii=False)

    print(f"Saving {output_path} done.")


if __name__ == "__main__":
    main()
