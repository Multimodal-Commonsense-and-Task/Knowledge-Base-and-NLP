import argparse
import glob
import json
from tqdm import tqdm

def merge_pq(dataset, generator, version):
    query_paths = sorted(glob.glob(f'data/{dataset}/query.{generator}.{version}.*.jsonl'), key=lambda x: int(x.split('.')[-2]))
    query_cnt = 0
    with open(f'data/{dataset}/query.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_query, \
        open(f'data/{dataset}/qrel.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_qrel:
        for query_path in tqdm(query_paths, desc='Merge pq'):
            with open(query_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    data = json.loads(line.strip())
                    did = data['did']
                    query = data['query']
                    query_cnt += 1
                    query_dump = json.dumps({'qid': f'genQ{query_cnt}', 'text': query})
                    qrel_dump = json.dumps({'qid': f'genQ{query_cnt}', 'did': did, 'score': 1})
                    fp_query.write(f'{query_dump}\n')
                    fp_qrel.write(f'{qrel_dump}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()

    merge_pq(args.dataset, args.generator, args.version)
