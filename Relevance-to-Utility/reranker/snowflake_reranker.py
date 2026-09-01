import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
import json
import torch
import pickle
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from snowflake.snowpark import Session
from snowflake.cortex import complete
import re

def extract_order_from_rerank_result(rerank_result):
    """
    Extracts the order of document IDs from the rerank result string.
    The expected format is like "[2] > [1] > [3] > [4]".
    """
    # Clean the result string - remove any unwanted characters and ensure we're working with expected format
    cleaned_result = rerank_result.strip()
    
    # Split by '>' to get individual document references
    parts = cleaned_result.split('>')
    
    order = []
    for part in parts:
        # Extract numbers between brackets
        part = part.strip()
        # Find the content between square brackets
        if '[' in part and ']' in part:
            start = part.find('[') + 1
            end = part.find(']')
            if start < end:
                try:
                    doc_id = int(part[start:end])
                    order.append(doc_id)
                except ValueError:
                    # Skip parts that don't convert to integers
                    raise ValueError(f"Invalid document ID in part: {part}")
    
    return order


def get_ranking_instruction(document, question, num):
    prompt = (
        "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.\n"
        f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {question}.\n"
        f"Passages\n{document}\n"
        f"Search Query: {question}.\n"
        f"Rank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance.\n"
        f"The output format should be [] > [], e.g., [4] > [2].\n"
        f"Only respond with the ranking results, do not say any word or explain.\n"
    )
    return prompt

def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pickle_dump(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def json_load(path):
    with open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def json_dump(path, data):
    with open(path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():

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

    # --------- Rerank ----------
    # Rank Zephyr model
    # reranker = ZephyrReranker(model_path = "models/rank_zephyr_7b_v1_full", vllm_batched=True,)
    # reranker = ZephyrReranker(model_path = "/data/models/rank_zephyr_7b_v1_full")
    dataset_name = 'hotpotqa' # crag

    if dataset_name == 'crag': # CRAG
        path = 'cache/crag'
        cache_path = 'search_cache.json'

    elif dataset_name == 'msmarco': # MSMARCO
        path = '../cache/msmarco'
        cache_path = 'search_cache_abs_500.json'

    elif dataset_name == 'hotpotqa': # HotpotQA
        path = 'cache/hotpotqa'
        cache_path = 'search_cache_b500_c500.json'

    cache = json_load(os.path.join(path, cache_path))
    
    new_cache = defaultdict(list)
    ranking_input_prompts = []
    for qid, (question, docs) in tqdm(enumerate(cache.items()), total=len(cache)):
        documents = ""
        for i, doc in enumerate(docs):
            if dataset_name == 'crag':
                if i >= 5:
                    break
            documents+= f"[{i}] {doc['contents']}\n\n"
        prompt = get_ranking_instruction(documents, question, len(docs))
        ranking_prompt = [{"role": "user", "content": prompt}]
        # ranking_input_prompts.append(ranking_prompt)

        retries = 3
        while retries > 0:
            try:
                response = complete(
                    model="claude-3-5-sonnet",
                    prompt=ranking_prompt,
                    options={
                        "max_tokens": 4096,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    }
                )
                reranked_result = extract_order_from_rerank_result(response)
                break
            except Exception as e:
                print(f"Error: {e}")
                retries -= 1
                if retries == 0:
                    reranked_result = [i for i in range(len(docs))]
                    print("Max retries reached, using default order.")
                    break
        new_cache[question] = [docs[i] for i in reranked_result]
        if qid < 3:
            print(f"Question: {question}")
            print(f"Reranked Result: {reranked_result}")
    
    # save the reranked cache
    output_path = os.path.join(path, cache_path.replace('.json', '_listwise_ranked_sonnet.json'))
    json_dump(output_path, new_cache)
    print(f'Saving {output_path} done.')



if __name__ == "__main__":
    main()