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
from glob import glob

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
        '--dataset_name', 
        type=str, 
        default='msmarco',
    )
    parser.add_argument(
        '--base_qa_path', 
        type=str, 
        default='/data/Search-o1/data/QA_Datasets',
    )
    parser.add_argument(
        '--raw_qa_file', 
        type=str, 
        default='msmarco_train_full.json'
    )
    parser.add_argument(
        '--base_cache_path', 
        type=str, 
        default='/data/Search-o1/cache'
    )
    parser.add_argument(
        '--raw_cache_file', 
        type=str, 
        default='search_cache_train_full.json'
    )
    parser.add_argument(
        '--train_dataset_dir', 
        type=str, 
        default='/data/FIRST/datasets'
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True,
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
    )
    parser.add_argument(
        '--qa_length',
        type=int,
        default=100000,
        help='Maximum number of questions to process from the QA dataset'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10
    )
    
    return parser.parse_args()


def detect_invalid_rewritten_doc(text):
    text = text.lower()
    if 'no_rewrite' in text:
        return True
    if 'no rewrite' in text:
        return True
    if len(text) == 0:
        return True
    return False

def LLM_rewrite_to_cache(args):

    def extract_contents(text):
        if 'Step 1. Document Rewriting:' not in text:
            return ""
        if 'Step 2. Answer:' not in text:
            return ""
        text = text.split('Step 1. Document Rewriting:')[-1]
        text = text.split('Step 2. Answer:')[0]
        text = text.strip()
        return text

    cache_path = os.path.join(args.base_cache_path, args.dataset_name)
    

    raw_cache_name = os.path.join(cache_path, args.raw_cache_file)
    raw_cache = json_load(raw_cache_name)

    cache_name = os.path.join(cache_path, args.input_file)
    if "*" in cache_name:
        cache_names = glob(cache_name)
    else:
        cache_names = [cache_name]
    print(f'cache_names: {cache_names}')
    new_cache = {}
    for cache_name in cache_names:
        cache = json_load(cache_name)    
        for question, docs in tqdm(cache.items()):
            try:
                raw_docs = raw_cache[question]
            except:
                if "mmlu" in args.dataset_name:
                    raw_docs = raw_cache.get(question.split("\n\n")[1], None)
            assert len(raw_docs[:args.top_k]) == len(docs[:args.top_k]), question

            new_docs = []
            for i, (raw_doc, doc) in enumerate(zip(raw_docs, docs)):
                assert raw_doc['id'] == doc['id']
                contents = extract_contents(doc['contents'])
                if detect_invalid_rewritten_doc(contents):
                    continue
                new_doc = deepcopy(doc)
                new_doc['position'] = i
                new_doc['contents'] = contents
                new_docs.append(new_doc)
            if len(new_docs) == 0:
                continue
            new_cache[question] = new_docs

    cache_name = "_".join(cache_name.split('_')[:-1]) if \
        cache_name.split("_")[-1].split(".")[0].isdigit() else cache_name[:-5]
    output_file = f'{cache_name}_split.json'
    json_dump(output_file, new_cache)
    print(f'new_cache: {len(new_cache)}')
    print(f'Convert {cache_name} -> {output_file}')

def cache_to_train_jsonl(args):

    def get_input_prompt(document, question):
        user_prompt = (
            'You are a helpful assistant. Your job is to analyze the documents below and rewrite only the parts that help clarify or refine the information in relation to the question.\n'
            'List each relevant document to better support answering the question. Do not include unrelated documents.\n\n'

            'Question:\n'
            f'{question}\n\n'
            
            'Documents:\n'
            f'{document}\n\n'
        )
        return user_prompt
    
    qa_path = os.path.join(args.base_qa_path, args.dataset_name)
    qa_name = os.path.join(qa_path, args.raw_qa_file)
    qa_list = json_load(qa_name)

    cache_path = os.path.join(args.base_cache_path, args.dataset_name)
    cache_name = os.path.join(cache_path, args.input_file)
    cache = json_load(cache_name)

    raw_cache_name = os.path.join(cache_path, args.raw_cache_file)
    raw_cache = json_load(raw_cache_name)
    # assert len(cache) == len(raw_cache), f'{args.raw_cache_file} and {args.input_file} should contain the same questions'
    
    qa_list = [qa for qa in qa_list if qa['Question'] in cache]
    data = []

    for qa in tqdm(qa_list):
        qid = qa['id']
        question = qa['Question']
        try:
            raw_docs = raw_cache[question]
        except:
            if "mmlu" in args.dataset_name:
                raw_docs = raw_cache.get(question.split("\n\n")[1], None)

        document = ""
        for i, doc in enumerate(raw_docs[:args.top_k]):
            document += f"Document {i + 1}:\n"
            document += f"{doc.get('contents', '')}\n\n"

        input_prompt = get_input_prompt(document, question)

        docs = cache[question]
        rewritten_document = ""
        rewritten_document += '<rewritten_docs>\n'
        for doc in docs[:args.top_k]:
            position = doc['position']
            rewritten_document += f"Document {position + 1}:\n"
            rewritten_document += f"{doc['contents']}\n\n"
        rewritten_document = rewritten_document.strip() + '\n'
        rewritten_document += '</rewritten_docs>\n'

        data.append({
            'id': qid,
            'conversations': [
                {'from': 'system', 'value': ''}, 
                {'from': 'human', 'id':qid, 'value': input_prompt}, 
                {'from': 'llama-3.3-70B-Instruct', 'value': rewritten_document}, 
            ]
        })
    if args.top_k != 10:
        output_file = os.path.join(args.train_dataset_dir, f'{args.dataset_name}_train_top{args.top_k}_tmp.jsonl')
    else:
        output_file = os.path.join(args.train_dataset_dir, f'{args.dataset_name}_train_tmp.jsonl')
    jsonl_dump(output_file, data)
    if args.top_k != 10:
        dev_output_file = os.path.join(args.train_dataset_dir, f'{args.dataset_name}_dev_top{args.top_k}_tmp.jsonl')
    else:
        dev_output_file = os.path.join(args.train_dataset_dir, f'{args.dataset_name}_dev_tmp.jsonl')
    jsonl_dump(dev_output_file, data[:100])
    print(f'data: {len(data)}')
    print(f'Convert {cache_name} -> {output_file}')


def SLM_rewrite_to_cache(args):

    def parse_doc_analysis(text: str) -> Dict[Union[int, Tuple[int, ...]], str]:
        # 1. Extract <doc_analysis>...</doc_analysis> block
        doc_analysis_match = re.search(r"<rewritten_docs>(.*?)</rewritten_docs>", text, re.DOTALL)
        if not doc_analysis_match:
            doc_analysis_match = doc_analysis_match = re.search(r"<rewritten_docs>(.*)", text, re.DOTALL)
            if not doc_analysis_match:
                return {}
        doc_analysis = doc_analysis_match.group(1)

        # 2. Match lines like "Document 1:", "Documents 2 and 3:", " Document 9:", etc.
        pattern = re.compile(
            r"\bDocument?\s+((?:\d+[,\s]*(?:and\s*)?)*)\s*:\s*(.*?)(?=\bDocuments?\s+\d|</doc_analysis>|\Z)",
            re.DOTALL
        )
        matches = pattern.findall(doc_analysis)
        if not matches: # without colons
            pattern = re.compile(
                r"\bDocuments?\s+((?:\d+[,\s]*(?:and\s*)?)*)\s*:?\s*\n(.*?)(?=\n\bDocuments?\s+\d|\Z)",
                re.DOTALL
            )
            matches = pattern.findall(doc_analysis)

        result = {}
        for doc_ids_str, content in matches:
            # Normalize: replace "and" with "," then split
            doc_ids_cleaned = doc_ids_str.replace("and", ",")
            doc_ids = tuple(sorted(set(int(n.strip()) for n in doc_ids_cleaned.split(",") if n.strip().isdigit())))
            key = doc_ids[0] if len(doc_ids) == 1 else doc_ids
            result[key] = content.strip()
        return result

    cache_path = os.path.join(args.base_cache_path, args.dataset_name)
    cache_name = os.path.join(cache_path, args.input_file)
    cache = json_load(cache_name)
    
    if args.dataset_name != 'gpqa':
        assert 'train' not in args.raw_cache_file, f'{args.raw_cache_file} should be test cache file'
    raw_cache_name = os.path.join(cache_path, args.raw_cache_file)
    raw_cache = json_load(raw_cache_name)
    assert len(cache) == len(raw_cache), f'{args.raw_cache_file} and {args.input_file} should contain the same questions'
    
    new_cache = {}
    failed_queries = []

    for question, docs in cache.items():
        assert len(docs) == 1
        docs = docs[0]['contents']
        doc_analysis = parse_doc_analysis(docs)
        
        raw_docs = raw_cache[question]
        new_docs = []
        
        for k, v in doc_analysis.items():
            if isinstance(k, tuple) or len(raw_docs) < k:
                doc = {'id':-1, 'contents':''}
            else:
                doc = deepcopy(raw_docs[k-1])

            if detect_invalid_rewritten_doc(v):
                continue
            doc['contents'] = v
            new_docs.append(doc)

        if len(new_docs) == 0:
            print(f'fail to rewrite for question: {question}')
            # print(f'docs: {docs}')
            failed_queries.append(question)
            new_docs = deepcopy(raw_docs)
        new_cache[question] = new_docs

    output_file = f'{cache_name[:-5]}_split.json'
    json_dump(output_file, new_cache)
    print(f'failed_queries: {len(failed_queries)}')
    print(f'new_cache: {len(new_cache)}')
    print(f'Convert {cache_name} -> {output_file}')



def main():
    args = parse_args()

    if args.mode == 'LLM_rewrite_to_cache':
        LLM_rewrite_to_cache(args)
    elif args.mode == 'cache_to_train_jsonl':
        cache_to_train_jsonl(args)
    elif args.mode == 'SLM_rewrite_to_cache':
        SLM_rewrite_to_cache(args)

if __name__ == "__main__":
    main()
