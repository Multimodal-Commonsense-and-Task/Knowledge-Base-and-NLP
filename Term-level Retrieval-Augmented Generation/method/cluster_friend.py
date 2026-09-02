import argparse
import json
from hamu_tool.dataset import DataLoader
from tqdm import tqdm

def cluster_friend(dataset : str) -> None:
    loader = DataLoader.load(f'beir/{dataset}')

    qrels = {}
    drels = {}
    for qrel in loader.get_qrels('test'):
        qid = qrel['qid']
        did = qrel['did']
        score = qrel['score']
        if score < 1:
            continue
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(did)
        if did not in drels:
            drels[did] = []
        drels[did].append(qid)

    avg_len = []
    with open(f'data/{dataset}/friend.jsonl', 'w', encoding='utf-8') as fp_friend:
        for i in tqdm(range(loader.total_docs()), desc=f'Clustering {dataset}'):
            did = loader.get_did(i)
            friends = set()
            if did in drels:
                for qid in drels[did]:
                    if len(friends) + len(qrels[qid]) > 100:
                        continue
                    friends.update(qrels[qid])
            friends = list(friends)
            if len(friends) == 0:
                friends = [did]
            avg_len.append(len(friends))
            dump = json.dumps({'did': did, 'friends': friends})
            fp_friend.write(f'{dump}\n')
    avg_len = sum(avg_len) / len(avg_len)
    print(f'Average length of friends: {avg_len:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    cluster_friend(args.dataset)
