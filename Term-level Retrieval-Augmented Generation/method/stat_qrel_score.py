import json
from hamu_tool.dataset import DataLoader
from hamu_tool.utils import CorpusReader

dataset = 'nfcorpus'

query_reader = CorpusReader(f'data/{dataset}/query.gpt35.base.idx')
loader = DataLoader.load(f'beir/{dataset}')
doc_reader = loader.reader_doc

score_min, score_max = -14, 14
scores = []
with open(f'data/{dataset}/sqrel.gpt35.base.jsonl', 'r', encoding='utf-8') as fp:
    for line in fp:
        data = json.loads(line)
        score = data['score']
        # score_min = min(score_min, score)
        # score_max = max(score_max, score)
        scores.append(score)
        if score < -10.0:
            qid = data['qid']
            query = query_reader[qid]['text']
            did = data['did']
            doc = (doc_reader[did]['title'] + ' ' + doc_reader[did]['text']).strip()
            print(f'[{qid}] {query}')
            print(f'[{did}] {doc}')
            input()

num_hist = 14
hist = [0 for _ in range(num_hist)]
for score in scores:
    idx = int((score - score_min) / (score_max - score_min) * num_hist)
    if idx >= num_hist:
        idx = num_hist
    if idx < 0:
        idx = 0
    hist[idx] += 1

print(f'Min: {score_min}, Max: {score_max}')

for idx, cnt in enumerate(hist):
    ratio = 100 * cnt / sum(hist)
    range_start = score_min + (score_max - score_min) / num_hist * idx
    range_end = score_min + (score_max - score_min) / num_hist * (idx + 1)
    print(f'{idx} ({range_start:.1f} ~ {range_end:.1f}): {ratio:.1f}')
