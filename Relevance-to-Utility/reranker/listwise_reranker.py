import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

from rank_llm.rerank.listwise import (
    ZephyrReranker,
)

@dataclass
class Query:
    text: str
    qid: Union[str | int]

@dataclass
class Candidate:
    docid: Union[str | int]
    score: float
    doc: Dict[str, Any]

@dataclass
class Request:
    query: Query
    candidates: List[Candidate] = field(default_factory=list)

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
    # --------- Rerank ----------
    # Rank Zephyr model
    if os.path.exists('models/rank_zephyr_7b_v1_full'):
        reranker = ZephyrReranker(model_path = "models/rank_zephyr_7b_v1_full")
    else:
        reranker = ZephyrReranker(model_path = "/data/models/rank_zephyr_7b_v1_full")
        
    dataset_name = 'mmlu'  # Change this to the dataset you want to rerank

    if dataset_name == 'crag': # CRAG
        path = '../cache/crag'
        cache_path = 'search_cache_500.json'

    elif dataset_name == 'msmarco': # MSMARCO
        path = '../cache/msmarco'
        cache_path = 'search_cache_ext_500.json'

    elif dataset_name == 'hotpotqa': # HotpotQA
        path = '../cache/hotpotqa'
        cache_path = 'search_cache_500.json'

    elif dataset_name == '2wiki': # 2wiki
        path = '../cache/2wiki'
        cache_path = 'search_cache_500.json'

    elif dataset_name == 'ambigqa': # musique
        path = '../cache/ambigqa'
        cache_path = 'search_cache_500.json'

    cache = json_load(os.path.join(path, cache_path))
    
    retrieved_results: List[Request] = []
    
    for qid, (question, docs) in tqdm(enumerate(cache.items())):
        documents = []
        for i, doc in enumerate(docs):
            documents.append(
                Candidate(
                    docid=doc.get('id', ''),
                    score=0.0,
                    doc={
                        'title': doc.get('title', ''),
                        'contents': doc.get('contents', ''),
                    }
                )
            )

        retrieved_results.append(
            Request(
                query=Query(text=question, qid=qid),
                candidates=documents
            )
        )
    
    rerank_results = reranker.rerank_batch(requests=retrieved_results, logging=False,
                                           rank_end = 10, window_size=10,
                                           top_k_retrieve=10)
    
    # print(f"Rerank results: {rerank_results}")
    new_cache = defaultdict(list)
    for result in rerank_results:
        question = result.query.text
        for candidate in result.candidates:
            doc = {
                'id': candidate.docid,
                'title': candidate.doc['title'],
                'contents': candidate.doc['contents'],
            }
            new_cache[question].append(doc)
    ranked_cache = new_cache
    # TODO: 
    output_path = os.path.join(path, cache_path.replace('.json', '_listwise_ranked.json'))
    json_dump(output_path, ranked_cache)
    print(f'Saving {output_path} done.')


if __name__ == "__main__":
    main()