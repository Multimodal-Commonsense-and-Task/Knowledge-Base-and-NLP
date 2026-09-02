import argparse
import json

def filter_scqrel(dataset : str, generator : str, version : str, threshold : float) -> None:
    with open(f'data/{dataset}/scqrel.{generator}.{version}.jsonl', 'r', encoding='utf-8') as fp_sqrel, \
        open(f'data/{dataset}/fscqrel.{generator}.{version}.jsonl', 'w', encoding='utf-8') as fp_fqrel:
        for line in fp_sqrel:
            data = json.loads(line.strip())
            qid = data['qid']
            did = data['did']
            score = data['score']
            if score >= threshold:
                dump = json.dumps({'qid': qid, 'did': did, 'score': score})
                fp_fqrel.write(f'{dump}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--generator', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()

    filter_scqrel(args.dataset, args.generator, args.version, args.threshold)
