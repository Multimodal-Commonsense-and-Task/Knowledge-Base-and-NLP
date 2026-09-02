import argparse
import json
import os
import re
from hamu_tool.dataset import DataLoader
from tqdm import tqdm
from utils.call_llm import LLM

def generate_pq(dataset : str, version : str, client_type : str, total_shard : int, shard : int) -> None:
    """Generates prompt-query pairs for the COT-v1 dataset.

    Args:
        dataset (str): Name of the dataset.
        version (str): Version of the prompt to generate. Valid options are: [cot-v1].
        client_type (str): Type of LLM client to use. Valid options are: [openai, azure].
        total_shard (int): Total number of shards splitted for parallel processing.
        shard (int): The shard to process.
    """
    llm = LLM(client_type=client_type)
    loader = DataLoader.load(f'beir/{dataset}')

    total_size = loader.total_docs()
    shard_size = (total_size + total_shard - 1) // total_shard
    shard_start = shard * shard_size
    shard_end = min((shard + 1) * shard_size, total_size)

    with open(f'data/{dataset}/prompt.generate.{version}.txt', 'r', encoding='utf-8') as fp:
        cot_prompt = fp.read().strip()

    did_set = set()
    qrels = loader.get_qrels('test')
    for qrel in qrels:
        if qrel['score'] <= 0:
            continue
        did_set.add(qrel['did'])

    query_split_regex = re.compile(r'\[\d+\]\s*([^\[\]]+)')

    cnt = 0
    err_cnt = 0
    pbar = tqdm(total=shard_end - shard_start, desc=f'SHARD {shard} : {shard_start} ~ {shard_end}')
    with open(f'data/{dataset}/query.gpt35.{version}.{shard}.jsonl', 'w', encoding='utf-8') as fp_query, \
        open(f'data/{dataset}/error.gpt35.{version}.{shard}.log', 'w', encoding='utf-8') as fp_log:
        for doc_idx in range(shard_start, shard_end):
            did = loader.get_did(doc_idx)
            if not did in did_set:
                cnt += 1
                pbar.update(1)
                continue
            doc = loader.get_doc(did)
            doc = (doc['title'] + ' ' + doc['text']).strip()
            prompt = cot_prompt.replace('##document##', doc)
            retry_cnt = 0
            while True:
                query_list = llm(prompt)
                if not query_list or retry_cnt >= 10:
                    fp_log.write(f'{did}\t{doc}\n')
                    err_cnt += 1
                else:
                    queries = query_split_regex.findall(query_list[0])
                    if len(queries) != 100:
                        retry_cnt += 1
                        continue
                    for query in queries:
                        query = ' '.join(query.split())
                        dump = json.dumps({'did': did, 'query': query})
                        fp_query.write(f'{dump}\n')
                break

            if cnt % 10 == 0:
                fp_query.flush()
                os.fsync(fp_query.fileno())
                fp_log.flush()
                os.fsync(fp_log.fileno())
            cnt += 1
            pbar.update(1)
    pbar.close()
    print('Error count:', err_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset.')
    parser.add_argument('--version', type=str, help='Version of the prompt to generate. Valid options are: [cot-v1].')
    parser.add_argument('--client_type', type=str, help='Type of LLM client to use. Valid options are: [openai, azure].')
    parser.add_argument('--total_shard', type=int, help='Total number of shards splitted for parallel processing.')
    parser.add_argument('--shard', type=int, help='The shard to process.')
    args = parser.parse_args()

    generate_pq(args.dataset, args.version, args.client_type, args.total_shard, args.shard)
