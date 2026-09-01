import os
import gc
import re
import json
import time
import torch
import random
import string
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional, Tuple
from utils.rouge import Rouge
from collections import Counter
from transformers import AutoTokenizer
rouge = Rouge()

from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import get_rejection_sampling_instruction


def truncate(tokenizer, max_dlen, text):
    text = text[:1024 * 128] # To speed up, truncate context with 128k character
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    if len(tokenized) > max_dlen:
        truncated = tokenized[:max_dlen]
        text = tokenizer.decode(truncated, skip_special_tokens=True)
    return text

def get_logprobs(output, prefix_len):
    assert output.prompt_logprobs is not None
    prompt_token_ids = output.prompt_token_ids[prefix_len:]
    prompt_logprobs = output.prompt_logprobs[prefix_len:]

    log_probs = []
    for token_id, token_logprobs in zip(prompt_token_ids, prompt_logprobs):
        if token_logprobs is None:
            continue
        assert token_id in token_logprobs, token_id

        log_prob = float(token_logprobs[token_id].logprob)
        decoded_token = token_logprobs[token_id].decoded_token
        log_probs.append(log_prob)
    return log_probs

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Name of the dataset to use."
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

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
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
        '--max_model_len',
        type=int,
        default=32768,
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=20480,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    parser.add_argument(
        '--max_dlen',
        type=int,
        default=2048,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    parser.add_argument(
        '--raw_cache_name',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--search_cache_name',
        type=str,
        required=True,
        help="Name of the cached search results name."
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help="Random seed for generation."
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
        '--debug',
        action='store_true',
    )
    return parser.parse_args()


def main():
    # Extract arguments
    args = parse_args()

    seed = args.seed
    mode = args.mode

    model_path = args.model_path
    subset_num = args.subset_num
    dataset_name = args.dataset_name
    subset_start_idx = args.subset_start_idx
    raw_cache_name = args.raw_cache_name
    search_cache_name = args.search_cache_name

    top_k = args.top_k
    top_p = args.top_p
    max_dlen = args.max_dlen
    max_tokens = args.max_tokens
    temperature = args.temperature
    max_model_len = args.max_model_len
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty

    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0

    print(f"#> args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Load dataset
    short_dataset_name = dataset_name.split("_")[0]
    data_path = f'./data/QA_Datasets/{short_dataset_name}/{dataset_name}.json'
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if subset_num != -1:
            data = data[args.subset_start_idx : args.subset_start_idx + subset_num]
    if args.debug:
        data = data[:10]
    print(f"#> Load qa: {data_path}; count: {len(data)}")

    cache_dir = f'./cache/{short_dataset_name}'
    raw_cache_path = os.path.join(cache_dir, raw_cache_name)
    search_cache_path = os.path.join(cache_dir, search_cache_name)
    with open(raw_cache_path, 'r', encoding='utf-8') as f:
        raw_cache = json.load(f)
    with open(search_cache_path, 'r', encoding='utf-8') as f:
        search_cache = json.load(f)
    print(f"#> Load raw_cache: {raw_cache_path}; count: {len(raw_cache)}")
    print(f"#> Load cache: {search_cache_path}; count: {len(search_cache)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'


    # Load model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        enforce_eager=True, 
        disable_log_stats=True, 
    )

    # Construct prompts
    input_prompts = []
    input_info = []

    for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
        qid = item['id']
        question = item['Question']
        answers = item['answer']
        answer = answers[0].strip()

        if question not in raw_cache:
            continue
        if question not in search_cache:
            continue

        # raw documents
        raw_docs = raw_cache[question]
        raw_docs = raw_docs[:top_k]
        docids = []
        doctitles = []
        formatted_documents = ""

        for i, doc_info in enumerate(raw_docs):
            docid = doc_info['id']
            doctitle = doc_info.get('title', '')
            contents = doc_info['contents']

            document = truncate(tokenizer, max_dlen, contents)
            document = f"Document {i+1}: {document}\n\n"

            docids.append(docid)
            doctitles.append(doctitle)
            formatted_documents += document
        formatted_documents = formatted_documents.strip()

        # rewritten documents
        rewritten_docs = search_cache[question]
        rewritten_docs = rewritten_docs[:top_k]
        formatted_rewritten_documents = ""

        for i, doc_info in enumerate(rewritten_docs):
            docid = doc_info['id']
            doctitle = doc_info.get('title', '')
            contents = doc_info['contents']

            document = truncate(tokenizer, max_dlen, contents)
            document = f"Rewritten Target Document {i+1}: {document}\n\n"

            docids.append(docid)
            doctitles.append(doctitle)
            formatted_rewritten_documents += document
        formatted_rewritten_documents = formatted_rewritten_documents.strip()

        user_prompt, assistent_prompt = get_rejection_sampling_instruction(
            question=question, 
            documents=formatted_documents, 
            rewrites=formatted_rewritten_documents, 
            num_of_docs=len(raw_docs), 
        )

        messages = []
        messages.append({"role": "user", "content": user_prompt})
        prefix_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_prompt += assistent_prompt
        prompt = prefix_prompt + "\n" + answer

        input_prompts.append(prompt)
        input_info.append([question, qid, answers, docids, doctitles, prefix_prompt])

    print(f"#> Total number of prompts: {len(input_prompts)}")


    # Generation
    print("#> Generation...")
    start_time = time.time()

    qa_list = []
    batch_size = len(input_prompts)

    for batch_idx in tqdm(range(0, len(input_prompts), batch_size)):
        batch_prompt = input_prompts[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        batch_info = input_info[batch_size * batch_idx : batch_size * (batch_idx + 1)]

        output_list = llm.generate(
            batch_prompt, 
            sampling_params=SamplingParams(
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k_sampling, 
                repetition_penalty=repetition_penalty,
                seed=seed,
                prompt_logprobs=0,
            )
        )
        for (question, qid, answers, docid, title, prefix_prompt), input_prompt, output in zip(batch_info, batch_prompt, output_list):
            prefix_ids = tokenizer(prefix_prompt, return_tensors="pt").input_ids
            prefix_len = prefix_ids.shape[1]

            log_probs = get_logprobs(output, prefix_len)

            qa_list.append({
                'id': qid, 
                'Question': question, 
                'docid': docid, 
                'answer': answers, 
                'log_probs': log_probs, 
                'avg_log_prob': round(np.mean(log_probs),4)
            })

        total_time = time.time() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")



    output_path = f'./data/QA_Datasets/{short_dataset_name}/answer_logprobs/{search_cache_name[:-5]}_answer_logprobs.json'
    with open(output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(qa_list, json_file, indent=4, ensure_ascii=False)
    print(f"Saving {output_path} done.")


if __name__ == "__main__":
    random.seed(42)
    main()
