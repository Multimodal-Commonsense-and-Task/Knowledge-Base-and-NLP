import re
import os
import time
import csv
import json
import torch
import random
import argparse
import numpy as np
import pickle

from tqdm import tqdm
from copy import deepcopy
from typing import Union, Tuple, Dict
from collections import defaultdict, Counter
from transformers import AutoTokenizer

from exit_rag import ExitRAG, Document


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
        

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Name of the dataset to use."
    )
    parser.add_argument(
        '--cache_name', 
        type=str, 
        default=None,
        help="HF datasets for documents to rerank"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    tokenizer = AutoTokenizer.from_pretrained('/data/models/Llama-3.1-8B-Instruct')
    truncate_doc_to = 1024

    cache_name = args.cache_name
    dataset_name = args.dataset_name
    short_dataset_name = dataset_name.split("_")[0]

    qa_path = f'/data/Search-o1/data/QA_Datasets/{short_dataset_name}/{dataset_name}.json'
    qa_list = json_load(qa_path)
    cache_path = f'/data/Search-o1/cache/{short_dataset_name}/{cache_name}.json'
    cache = json_load(cache_path)

    # Initialize pipeline
    rag = ExitRAG(
        retriever_model="/data/models/gemma-2b-it",
        compression_model="/data/models/exit-gemma-2b",
        reader_model="/data/models/Llama-3.1-8B-Instruct"
    )

    new_cache = {}
    for qa in tqdm(tqdm(qa_list[:3])):
        question = qa['Question']
        docs = cache[question]

        if truncate_doc_to is not None:
            truncated_docs = []
            for doc in docs:
                truncated_doc = deepcopy(doc)
                contents = truncated_doc['contents']
                contents = tokenizer.decode(tokenizer.encode(contents)[:truncate_doc_to])
                truncated_doc['contents'] = contents
                truncated_docs.append(truncated_doc)
            docs = truncated_docs

        documents = [Document(title=doc.get('title',''), text=doc['contents']) for doc in docs]

        compressed_text, selections, scores = rag.compress_documents(
            query=question,
            documents=documents,
            threshold=0.5  # Adjustable compression threshold
        )
        compressed_text = compressed_text.strip()

        if len(compressed_text) == 0:
            new_cache[question] = deepcopy(docs)
        else:
            new_cache[question] = [{'id':-1, 'contents':compressed_text}]


    output_path = f'../../cache/{short_dataset_name}/{cache_name}_exit.json'
    json_dump(output_path, new_cache)
    print(f'Saving {output_path} done.')





if __name__ == "__main__":
    main()