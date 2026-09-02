import argparse
import json
import os
from hamu_tool.dataset import DataLoader

def export_gpl(dataset : str, generator : str, version : str) -> None:
    if not os.path.exists(f'gpl/{dataset}'):
        os.makedirs(f'gpl/{dataset}')

    loader = DataLoader.load(f'beir/{dataset}')

    with open(f'gpl/{dataset}/corpus.jsonl', 'w', encoding='utf-8') as fp_corpus:
        for doc in loader.get_docs():
            dump = json.dumps({'_id': doc['id'], 'text': (doc['title'] + ' ' + doc['text']).strip(), 'title': doc['title']})
            fp_corpus.write(f'{dump}\n')

    with open(f'data/{dataset}/fquery.{generator}.{version}.jsonl', 'r', encoding='utf-8') as fp_gen, \
        open(f'gpl/{dataset}/qgen-queries.jsonl', 'w', encoding='utf-8') as fp_gpl:
        for line in fp_gen:
            data = json.loads(line.strip())
            dump = json.dumps({'_id': data['qid'], 'text': data['text'], 'metadata': {}})
            fp_gpl.write(f'{dump}\n')

    with open(f'data/{dataset}/fscqrel.{generator}.{version}.jsonl', 'r', encoding='utf-8') as fp_gen, \
        open(f'gpl/{dataset}/train.tsv', 'w', encoding='utf-8') as fp_gpl:
        fp_gpl.write('query-id\tcorpus-id\tscore\n')
        for line in fp_gen:
            data = json.loads(line.strip())
            qid = data['qid']
            did = data['did']
            fp_gpl.write(f'{qid}\t{did}\t1\n')

    with open(f'gpl/{dataset}/queries.jsonl', 'w', encoding='utf-8') as fp_gpl:
        for query in loader.get_queries():
            dump = json.dumps({'_id': query['id'], 'text': query['text'], 'metadata': {}})
            fp_gpl.write(f'{dump}\n')

    with open(f'gpl/{dataset}/test.tsv', 'w', encoding='utf-8') as fp_gpl:
        fp_gpl.write('query-id\tcorpus-id\tscore\n')
        for qrel in loader.get_qrels('test'):
            qid = qrel['qid']
            did = qrel['did']
            score = qrel['score']
            fp_gpl.write(f'{qid}\t{did}\t{score}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()

    export_gpl(args.dataset, args.generator, args.version)
