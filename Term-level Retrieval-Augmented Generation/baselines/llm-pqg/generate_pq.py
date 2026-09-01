import argparse
import json
import os
import ray
import re
from hamu_tool.dataset import DataLoader
from ray.experimental.tqdm_ray import tqdm
from utils.call_llm import LLM

@ray.remote
class ProgressActor:
    def __init__(self, total, desc=''):
        self.progress = tqdm(total=total, desc=desc)

    def update(self):
        self.progress.update(1)

    def close(self):
        self.progress.close()

def generate_pq(dataset : str, generator : str, client_type : str, total_shard : int) -> None:
    if not os.path.exists(f'data/{dataset}'):
        os.makedirs(f'data/{dataset}')

    if generator == 'gpt35':
        loader = DataLoader.load(f'beir/{dataset}')
        total_size = loader.total_docs()
        progress_actor = ProgressActor.remote(total_size, desc='Generating PQs')
        ret_ids = []
        for shard in range(total_shard):
            ret_ids.append(generate_pq_gpt35.remote(dataset, client_type, total_shard, shard, progress_actor))
        err_list = ray.get(ret_ids)
        ray.get(progress_actor.close.remote())
        for i in range(len(err_list)):
            if err_list[i] > 0:
                print(f'Error count for shard {i}: {err_list[i]}')
    else:
        raise ValueError(f'Invalid generator: [{generator}]. Valid options are: [gpt35]')

@ray.remote
def generate_pq_gpt35(dataset : str, client_type : str, total_shard : int, shard : int, progress_actor : ProgressActor) -> None:
    llm = LLM(client_type=client_type)
    loader = DataLoader.load(f'beir/{dataset}')

    total_size = loader.total_docs()
    shard_size = (total_size + total_shard - 1) // total_shard
    shard_start = shard * shard_size
    shard_end = min((shard + 1) * shard_size, total_size)

    with open(f'data/{dataset}/prompt.gpt35.base.txt', 'r', encoding='utf-8') as fp:
        prompt_base = fp.read().strip()

    did_set = set()
    qrels = loader.get_qrels('test')
    for qrel in qrels:
        if qrel['score'] <= 0:
            continue
        did_set.add(qrel['did'])

    query_split_regex = re.compile(r'\[\d+\]\s*([^\[\]]+)')

    cnt = 0
    err_cnt = 0
    with open(f'data/{dataset}/query.gpt35.base.{shard}.jsonl', 'w', encoding='utf-8') as fp_query, \
        open(f'data/{dataset}/error.gpt35.base.{shard}.log', 'w', encoding='utf-8') as fp_log:
        for doc_idx in range(shard_start, shard_end):
            did = loader.get_did(doc_idx)
            if not did in did_set:
                cnt += 1
                ray.get(progress_actor.update.remote())
                continue
            doc = loader.get_doc(did)
            doc = (doc['title'] + ' ' + doc['text']).strip()
            prompt = prompt_base.replace('##document##', doc)
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
            ray.get(progress_actor.update.remote())
    return err_cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset.')
    parser.add_argument('--generator', type=str, help='Version of the prompt to generate. Valid options are: [gpt35].')
    parser.add_argument('--client_type', type=str, help='Type of LLM client to use. Valid options are: [openai, azure].')
    parser.add_argument('--total_shard', type=int, help='Total number of shards splitted for parallel processing.')
    args = parser.parse_args()

    ray.init(dashboard_host='0.0.0.0', num_cpus=32)
    generate_pq(args.dataset, args.generator, args.client_type, args.total_shard)
