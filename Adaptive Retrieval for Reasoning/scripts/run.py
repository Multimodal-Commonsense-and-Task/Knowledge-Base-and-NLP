import re
import os
import sys
import csv
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Tuple, Dict, List
from collections import defaultdict, Counter

from scripts.parser import parse_args
from utils.utils import json_load, json_dump


### Ranking
def rerank(args):
    from scripts.reranker import Reranker
    print(f"#> rerank args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Load data
    cache = json_load(args.cache_path)

    model_short_name = args.model_name_or_path.split("/")[-1].lower()
    output_path = args.cache_path[:-5] + f"_{args.prompt_mode}_{model_short_name}.json"

    if args.max_dlen != 1024:
        output_path = output_path[:-5] + f"_{args.max_dlen}.json"
    if args.peft_model_name is not None:
        short_peft_model_name = "_".join(args.peft_model_name.split("/")[-2:]).lower()
        output_path = output_path[:-5] + f"_{short_peft_model_name}.json"
    if args.reuse:
        output_path = output_path[:-5] + f"_reuse.json"
    if args.d2d:
        output_path = output_path[:-5] + f"_{args.d2d}.json"
    if args.alpha != 1.0:
        output_path = output_path[:-5] + f"_alpha{args.alpha}.json"
    if args.beta != 1.0:
        output_path = output_path[:-5] + f"_beta{args.beta}.json"
    if args.bt_iter != 5:
        output_path = output_path[:-5] + f"_btiter{args.bt_iter}.json"
    if args.seed != 0:
        output_path = output_path[:-5] + f"_seed{args.seed}.json"
    if args.slidegar != 0:
        output_path = output_path[:-5] + f"_slidegar.json"

    args.output_path = output_path
    print(f"#> output_path: {output_path}")

    ranked_cache = {}
    # tmp_path = output_path[:-5] + "_tmp.json"
    # if os.path.exists(tmp_path):
    #     ranked_cache = json_load(tmp_path)
    #     print(f"#> Load tmp file; {tmp_path}; ranked_cache: {len(ranked_cache)}/{len(cache)}")
    args.ranked_cache = ranked_cache

    # Prepare data
    queries, documents = [], []
    for query, docs in cache.items():
        if query in ranked_cache:
            continue
        if 'contents' in docs[0]:
            docs = [{'id':doc['id'], 'content':doc['contents']} for doc in docs]
        queries.append(query)
        documents.append(docs[:args.top_k_docs])

    # Rerank
    reranker = Reranker(**args.__dict__)
    results = reranker.rerank(queries, documents)

    # Saving results
    for query, docs, ranked_docs in zip(queries, documents, results):
        if len(ranked_docs) == 0:
            print(f"#> Empty docs for query {query}")
            ranked_docs = docs
        ranked_cache[query] = ranked_docs
    
    if args.vllm or args.snowflake:
        json_dump(output_path, ranked_cache)
        print(f"#> Saving {output_path} done\n")
    else:
        if args.local_rank == 0:
            json_dump(output_path, ranked_cache)
            print(f"#> Saving {output_path} done in args.local_rank: {args.local_rank}\n")


### Generate training dataset
def generate_data(args):
    from scripts.tracer import Tracer

    print(f"#> generate_data args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    dir_path = os.path.dirname(args.cache_path)
    model_short_name = args.model_name_or_path.split("/")[-1].lower()
    output_file = os.path.basename(args.cache_path).replace("search_cache", "trace_cache")[:-5] + f"_{args.prompt_mode}_{model_short_name}.json"
    output_path = os.path.join(dir_path, output_file)
    print(f"#> output_path: {output_path}")

    cache = json_load(args.cache_path)

    trace_cache = {}
    tmp_path = output_path[:-5] + "_tmp.json"
    # if os.path.exists(tmp_path):
    #     trace_cache = json_load(tmp_path)
    #     print(f"#> Load tmp file; {tmp_path}; trace_cache: {len(trace_cache)}/{len(cache)}")

    # Prepare data
    queries, documents = [], []
    if isinstance(cache, list):
        for data in cache:
            query = data['query']
            docs = data['docs']
            queries.append(query)
            documents.append(docs[:args.top_k_docs])
    elif isinstance(cache, dict):
        for query, docs in cache.items():
            if query in trace_cache:
                continue
            queries.append(query)
            documents.append(docs[:args.top_k_docs])

    tracer = Tracer(**args.__dict__)
    results = tracer.generate_data(queries, documents)

    trace_cache_list = []
    for query, result in zip(queries, results):
        if len(result['trace']) == 0:
            continue
        if len(result['docs']) == 0:
            continue
        trace_cache[query] = result
        result['query'] = query
        trace_cache_list.append(result)

    # json_dump(output_path, trace_cache)
    if isinstance(cache, list):
        json_dump(output_path, trace_cache_list)
    elif isinstance(cache, dict):
        json_dump(output_path, trace_cache)
    print(f"#> Saving {output_path} done\n")


### Train
def train_dpo(args):
    from scripts.trainer_dpo import Trainer
    from accelerate import Accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"#> train_dpo args")
        for k, v in vars(args).items():
            print(f"{k}: {v}")

    trainer = Trainer(accelerator, **args.__dict__)
    trainer.train()


def train_sft(args):
    from scripts.trainer_sft import Trainer
    print(f"#> train_sft args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    trainer = Trainer(**args.__dict__)
    trainer.train()


def train_grpo(args):
    from scripts.trainer_grpo import Trainer
    print(f"#> train_grpo args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    trainer = Trainer(**args.__dict__)
    trainer.train()


### Construct graph
def construct_graph(args):
    from scripts.graph import build_graph
    print(f"#> construct_graph args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    build_graph(args)


def sunar(args):
    from scripts.sunar import SUNAR
    print(f"#> sunar args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Load data
    qa_list = json_load(args.qa_path)
    cache = json_load(args.cache_path)

    model_short_name = args.model_name_or_path.split("/")[-1].lower()
    output_path = args.cache_path[:-5] + f"_sunar_{args.prompt_mode}_{model_short_name}.json"

    if args.peft_model_name is not None:
        short_peft_model_name = "_".join(args.peft_model_name.split("/")[-2:]).lower()
        output_path = output_path[:-5] + f"_{short_peft_model_name}.json"
    if args.reuse:
        output_path = output_path[:-5] + f"_reuse.json"
    if args.d2d:
        output_path = output_path[:-5] + f"_{args.d2d}.json"

    args.output_path = output_path
    print(f"#> output_path: {output_path}")

    results = []
    # tmp_path = output_path[:-5] + "_tmp.json"
    # if os.path.exists(tmp_path):
    #     results = json_load(tmp_path)
    #     print(f"#> Load tmp file; {tmp_path}; ranked_cache: {len(results)}/{len(qa_list)}")
    args.tmp_results = results

    # For skip
    qids_done = set()
    for qa in results:
        qid = qa['id']
        qids_done.add(qid)

    qa_data = []
    for qa in qa_list:
        qid = qa['id']
        if qid in qids_done:
            continue

        if 'Question' in qa:
            query = qa['Question']
        elif 'query' in qa:
            query = qa['query']

        if 'answer' in qa:
            answer = qa['answer']
        elif 'gold_answer' in qa:
            answer = qa['gold_answer']
        if isinstance(answer, str):
            answer = [answer]

        docs = cache[query]
        if 'contents' in docs[0]:
            docs = [{'id':doc['id'], 'content':doc['contents']} for doc in docs]
        qa_data.append({
            'id':qid, 
            'query':query, 
            'answer':answer, 
            'docs':docs[:args.top_k_docs]
        })
    
    args.gen_final_answer = "bright" not in args.qa_path

    sunar = SUNAR(**args.__dict__)
    results = sunar.inference(qa_data, cache)
    json_dump(output_path, results)
    print(f"#> Saving {output_path} done\n")

    ranked_cache = {result['query']: result['final_docs'] for result in results}
    output_path = output_path[:-5] + "_final_docs.json"
    json_dump(output_path, ranked_cache)
    print(f"#> Saving {output_path} done\n")



def main():
    args = parse_args()
    
    if args.task == 'rerank':
        rerank(args)
    elif args.task == 'generate_data':
        generate_data(args)
    elif args.task == 'train_dpo':
        train_dpo(args)
    elif args.task == 'train_sft':
        train_sft(args)
    elif args.task == 'train_grpo':
        train_grpo(args)
    elif args.task == 'construct_graph':
        construct_graph(args)
    elif args.task == 'sunar':
        sunar(args)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
