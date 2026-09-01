
import re
import os
import time
import csv
import json
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from typing import Union, Tuple, Dict
from collections import defaultdict, Counter
from transformers import AutoModel

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

def jsonl_load(path):
    data = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def jsonl_dump(path, data):
    with open(path, mode='w', encoding='utf-8') as f:
        for qd in data:
            f.write(f"{json.dumps(qd, ensure_ascii=False)}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Run format conversion from chatgpt output")
    
    parser.add_argument(
        '--base_cache_path', 
        type=str, 
        default='../cache'
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='msmarco',
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='search_cache_abs_500.json',
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='/data/models/provence-reranker-debertav3-v1',
    )

    return parser.parse_args()


def pruning(args):
    provence = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    provence.eval()
    provence.cuda()


    cache_path = os.path.join(args.base_cache_path, args.dataset_name)
    cache_name = os.path.join(cache_path, args.input_file)
    cache = json_load(cache_name)

    new_cache = {}
    for question, docs in cache.items():
        contents = [[doc['contents'] for doc in docs]]
        provence_output = provence.process([question], contents)

        reranking_scores = provence_output['reranking_score'][0]
        pruned_docs = provence_output['pruned_context'][0]
        assert len(reranking_scores) == len(pruned_docs)

        indices = [i for i in range(len(pruned_docs))]
        indices = sorted(indices, key=lambda x: reranking_scores[x], reverse=True)

        reranking_scores = [reranking_scores[i] for i in indices]
        pruned_docs = [pruned_docs[i] for i in indices]
        assert len(pruned_docs) == len(docs)

        new_docs = []
        for doc, pruned_doc in zip(docs, pruned_docs):
            if len(pruned_doc) == 0:
                continue
            doc['contents'] = pruned_doc
            new_docs.append(deepcopy(doc))
        
        if len(new_docs) == 0:
            print(f'failed to prune for question: {question}')
            new_docs = deepcopy(docs)
        new_cache[question] = new_docs


    output_file = f'{cache_name[:-5]}_provence.json'
    json_dump(output_file, new_cache)
    print(f'new_cache: {len(new_cache)}')
    print(f'Convert {cache_name} -> {output_file}')
        

def main():
    args = parse_args()

    pruning(args)


if __name__ == "__main__":
    main()
