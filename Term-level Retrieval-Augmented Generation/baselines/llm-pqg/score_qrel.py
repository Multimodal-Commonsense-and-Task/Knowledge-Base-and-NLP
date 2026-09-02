import argparse
import json
import os
import ray
from hamu_tool.dataset import DataLoader
from hamu_tool.utils import CorpusReader
from sentence_transformers import CrossEncoder
from utils import get_total_line, ProgressActor

def score_qrel(dataset : str, generator : str, gpus : str) -> None:
    total_size = get_total_line(f'data/{dataset}/query.{generator}.base.jsonl')
    pbar = ProgressActor.remote(total_size, desc='Scoring qrel')
    gpus = [ int(gpu.strip()) for gpu in gpus.split(',')]
    ret_ids = []
    for i in range(len(gpus)):
        ret_ids.append(score_qrel_shard.remote(dataset, generator, len(gpus), i, gpus[i], pbar))
    ray.get(ret_ids)
    ray.get(pbar.close.remote())

@ray.remote
def score_qrel_shard(dataset : str, generator : str, total_shard : int, shard : int, gpu : int, pbar : ProgressActor) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    doc_reader = DataLoader.load(f'beir/{dataset}')
    query_reader = CorpusReader(f'data/{dataset}/query.{generator}.base.idx')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    total_size = get_total_line(f'data/{dataset}/query.{generator}.base.jsonl')
    shard_size = (total_size + total_shard - 1) // total_shard
    shard_start = shard * shard_size
    shard_end = min((shard + 1) * shard_size, total_size)

    cnt = 0
    real_cnt = 0
    with open(f'data/{dataset}/qrel.{generator}.base.jsonl', 'r', encoding='utf-8') as fp_qrel, \
        open(f'data/{dataset}/qrel.{generator}.score.{shard}.jsonl', 'w', encoding='utf-8') as fp_sqrel:
        for line in fp_qrel:
            if cnt < shard_start:
                cnt += 1
                continue
            if cnt >= shard_end:
                break

            data = json.loads(line.strip())
            query = query_reader[data['qid']]['text']
            doc = doc_reader.get_doc(data['did'])
            doc = (doc['title'] + ' ' + doc['text']).strip()
            score = cross_encoder.predict([query, doc]).tolist()
            dump = json.dumps({'qid': data['qid'], 'did': data['did'], 'score': score})
            fp_sqrel.write(f'{dump}\n')

            if real_cnt % 100 == 0:
                fp_sqrel.flush()
                os.fsync(fp_sqrel.fileno())
            cnt += 1
            real_cnt += 1
            ray.get(pbar.update.remote())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--gpus', type=str, required=True)
    args = parser.parse_args()

    ray.init(dashboard_host='0.0.0.0', num_cpus=32)
    score_qrel(args.dataset, args.generator, args.gpus)
