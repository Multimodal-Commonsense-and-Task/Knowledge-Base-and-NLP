import re
import os
import csv
import time
import json
import torch
import random
import pickle
import argparse
import pytrec_eval
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

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

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    # oracle reranker evaluation
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), key=lambda x: results[query_id][x], reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if doc_id in top_100_ids[query_id] and query_id in qrels and doc_id in qrels[query_id]: # a doc is both top 100 and also in ground truth
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id] # extract the score from ground truth
            else:
                oracle_results[query_id][doc_id] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    oracle_scores = evaluator.evaluate(oracle_results)
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg}
    return output


def main(args):
    # qrels
    examples = json_load(args.qa_file)
    
    key = 'gold_ids'
    ground_truth = {}
    question2qid = {}
    qid2excluded_ids = defaultdict(set)
    for e in examples:
        question = e['query']
        assert question not in question2qid
        question2qid[question] = e['id']

        ground_truth[e['id']] = {}
        if isinstance(e[key], dict):
            for gid, rel in e[key].items():
                ground_truth[e['id']][gid] = rel
        else:
            for gid in e[key]:
                ground_truth[e['id']][gid] = 1

        if 'excluded_ids' in e:
            for did in e['excluded_ids']:
                qid2excluded_ids[e['id']].add(did)
                assert not did in ground_truth[e['id']]

    # base_cache = json_load(args.base_cache_file)

    # results
    cache = json_load(args.cache_file)
    invalid_queries = []
    # invalid_queries = json_load("tmp_invalid_queries.json")

    if isinstance(cache, list):
        cache = {data['query']:data['final_docs'] for data in cache}
    
    scores = {}
    for question, docs in cache.items():
        if len(docs) < 10:
            invalid_queries.append(question)
            doc_ids = set([doc['id'] for doc in docs])
            # base_docs = base_cache[question]
            # docs += [doc for doc in base_docs if doc['id'] not in doc_ids]

        qid = question2qid[question]
        assert qid not in scores
        scores[qid] = {}

        for rank, doc in enumerate(docs[:100]):
            did = doc['id']
            assert did not in qid2excluded_ids[qid]
            scores[qid][did] = 1 / (rank + 1)
    
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    print(f"#> invalid_queries: {len(invalid_queries)}")
    print(f"#> scores: {len(scores)}/{len(qid2excluded_ids)}")
    print(f"#> NDCG@10: {round(100*results['NDCG@10'],1)}\tRecall@10: {round(100*results['Recall@10'],1)}\tP@10: {round(100*results['P@10'],1)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run format conversion from chatgpt output")
    parser.add_argument(
        '--qa_file', 
        type=str, 
        default='/data/Search-o1/data/QA_Datasets/bright/aops.json',
    )
    parser.add_argument(
        '--cache_file', 
        type=str, 
        default='/data/Search-o1/cache/bright/search_cache_bm25_aops.json',
    )
    parser.add_argument(
        '--base_cache_file', 
        type=str, 
        default='/data/Search-o1/cache/bright/search_cache_bm25.json',
    )
    args = parser.parse_args()

    main(args)