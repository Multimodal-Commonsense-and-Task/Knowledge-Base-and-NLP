import argparse
import glob
import json
from tqdm import tqdm

def merge_scqrel(dataset, generator, version):
    query_paths = sorted(glob.glob(f'data/{dataset}/sfqrel.{generator}.{version}.*.jsonl'), key=lambda x: int(x.split('.')[-2]))
    with open(f'data/{dataset}/scqrel.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_qrel:
        for query_path in tqdm(query_paths, desc='Merge scqrel'):
            with open(query_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    data = json.loads(line.strip())
                    qid = data['qid']
                    did = data['did']
                    score = data['score']
                    dump = json.dumps({'qid': qid, 'did': did, 'score': score})
                    fp_qrel.write(f'{dump}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()

    merge_scqrel(args.dataset, args.generator, args.version)
