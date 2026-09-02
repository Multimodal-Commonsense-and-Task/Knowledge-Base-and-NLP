import argparse
import glob
import json
import logging
import os
import multiprocessing as mp

from src.util.const import *
from src.util.dtype import Query
from src.method import BM25, EMR, SMR

def generate_tag(args: argparse.Namespace) -> str:
    tags = []
    match args.method:
        case 'bm25':
            tags.extend([f'doc{args.doc_topk}', f'k{args.bm25_k1}', f'b{args.bm25_b}'])
        case 'reasonir':
            tags.extend([f'doc{args.doc_topk}'])
        case 'smr':
            tags.extend([f'{args.llm}', f'{args.retriever}', f'doc{args.doc_topk}'])
        case 'emr':
            tags.extend([f'{args.llm}', f'{args.retriever}', f'doc{args.doc_topk}', f'sent{args.sent_topk}'])
    tag = '_'.join(tags)
    return tag

def validate_idx(args: argparse.Namespace, tag: str):
    fnames = glob.glob(f'output/log/{args.dataset}-{args.query_type}/{args.method}.{tag}.*.log')
    last_idx = -1
    is_duplicated_idx = False
    for fname in fnames:
        find_idx = int(fname.split('.')[-2])
        last_idx = max(last_idx, find_idx)
        if find_idx == args.idx:
            is_duplicated_idx = True
    if is_duplicated_idx:
        raise ValueError(f'Index [{args.idx}] is duplicated, please use a different index (last used index is [{last_idx}])')

def load_queries(args: argparse.Namespace) -> list[Query]:
    queries = []
    with open(f'data/{args.dataset}/query.{args.query_type}.jsonl', 'r', encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            query = Query(
                qid=data['qid'],
                text=data['text'],
                pos_dids=set(data['pos_dids']),
                pos_dids_long=set(data['pos_dids_long']),
                excluded_dids=set(data['excluded_dids']),
                gold_answer=data['gold_answer'],
                reasoning=data['reasoning']
            )
            queries.append(query)
    return queries

def main(args: argparse.Namespace):
    os.makedirs(f'output/history/{args.dataset}-{args.query_type}', exist_ok=True)
    os.makedirs(f'output/evaluation/{args.dataset}-{args.query_type}', exist_ok=True)
    os.makedirs(f'output/log/{args.dataset}-{args.query_type}', exist_ok=True)

    tag = generate_tag(args)
    validate_idx(args, tag)
    file_name = f'{args.method}.{tag}.{args.idx}'
    logging.basicConfig(filename=f'output/log/{args.dataset}-{args.query_type}/{file_name}.log', level=logging.DEBUG, force=True)

    queries = load_queries(args)
    model = METHOD_MAPPING[args.method](args=args, tag=tag)
    model.run(queries=queries, k=args.doc_topk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SMR with Qwen and ReasonIR Retriever')
    parser.add_argument('--dataset', type=str, default='biology', help='Dataset to use (default: biology)', choices=VALID_DATASETS)
    parser.add_argument('--query_type', type=str, default='original', help='Type of queries to use (default: original)', choices=VALID_QUERY_TYPES)
    parser.add_argument('--method', type=str, default='emr', help='Method to use (default: emr)', choices=VALID_METHODS)
    parser.add_argument('--llm', type=str, default='qwen3', help='LLM model to use (default: qwen3)', choices=VALID_LLMS)
    parser.add_argument('--retriever', type=str, default='bm25', help='Retriever to use (default: bm25)', choices=VALID_RETRIEVERS)
    parser.add_argument('--max_steps', type=int, default=16, help='Maximum number of reasoning steps (default: 16)')
    parser.add_argument('--doc_topk', type=int, default=10, help='Number of top documents to retrieve (default: 10)')
    parser.add_argument('--sent_topk', type=int, default=5, help='Number of top sentences to retrieve (default: 5)')
    parser.add_argument('--init_temp', type=float, default=0.1, help='Initial temperature for LLM (default: 0.1)')
    parser.add_argument('--bm25_k1', type=float, default=0.9, help='BM25 k1 parameter (default: 0.9)')
    parser.add_argument('--bm25_b', type=float, default=0.2, help='BM25 b parameter (default: 0.2)')
    parser.add_argument('--idx', type=int, default=0, help='Index of the current run (default: 0)')
    parser.add_argument('--port', type=int, default=8000, help='Port for LLM api (default: 8000)')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    print(args)
    main(args)
