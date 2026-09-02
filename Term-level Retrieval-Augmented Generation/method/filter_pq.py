import argparse
import json
from hamu_tool.utils import CorpusReader

def filter_pq(dataset : str, generator : str, version : str, threshold : float) -> None:
    reader = CorpusReader(f'data/{dataset}/query.{generator}.{version}.idx')
    with open(f'data/{dataset}/sqrel.{generator}.{version}.jsonl', 'r', encoding='utf-8') as fp_sqrel, \
        open(f'data/{dataset}/fquery.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_fquery, \
        open(f'data/{dataset}/fqrel.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_fqrel:
        for line in fp_sqrel:
            data = json.loads(line.strip())
            query = reader[data['qid']]['text']
            if data['score'] >= threshold:
                dump = json.dumps({'qid': data['qid'], 'text': query})
                fp_fquery.write(f'{dump}\n')
                fp_fqrel.write(f'{line.strip()}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()

    filter_pq(args.dataset, args.generator, args.version, args.threshold)
