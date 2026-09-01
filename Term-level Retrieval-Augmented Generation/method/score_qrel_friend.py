import argparse
import json
import os
from hamu_tool.dataset import DataLoader
from hamu_tool.utils import CorpusReader
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from utils import get_total_line

def score_qrel_friend(dataset : str, generator : str, version : str, total_shard : int, shard : int) -> None:
    friends = {}
    with open(f'data/{dataset}/friend.jsonl', 'r', encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            friends[data['did']] = data['friends']

    doc_reader = DataLoader.load(f'beir/{dataset}')
    query_reader = CorpusReader(f'data/{dataset}/query.{generator}.{version}.idx')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    total_size = get_total_line(f'data/{dataset}/fqrel.{generator}.{version}.jsonl')
    shard_size = (total_size + total_shard - 1) // total_shard
    shard_start = shard * shard_size
    shard_end = min((shard + 1) * shard_size, total_size)

    cnt = 0
    real_cnt = 0
    pbar = tqdm(total=shard_end - shard_start, desc=f'SHARD {shard} : {shard_start} ~ {shard_end}')
    with open(f'data/{dataset}/fqrel.{generator}.{version}.jsonl', 'r', encoding='utf-8') as fp, \
        open(f'data/{dataset}/scqrel.{generator}.{version}.{shard}.jsonl', 'w', encoding='utf-8') as fp_scqrel:
        for line in fp:
            if cnt < shard_start:
                cnt += 1
                continue
            if cnt >= shard_end:
                break

            data = json.loads(line)
            qid = data['qid']
            did = data['did']
            query = query_reader[qid]['text']
            for friend in friends[did]:
                doc = doc_reader.get_doc(friend)
                doc = (doc['title'] + ' ' + doc['text']).strip()
                score = cross_encoder.predict([query, doc]).tolist()
                dump = json.dumps({'qid': data['qid'], 'did': friend, 'score': score})
                fp_scqrel.write(f'{dump}\n')

            if real_cnt % 100 == 0:
                fp_scqrel.flush()
                os.fsync(fp_scqrel.fileno())
            cnt += 1
            real_cnt += 1
            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--total_shard', type=int, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    score_qrel_friend(args.dataset, args.generator, args.version, args.total_shard, args.shard)
