from hamu_tool.dataset import DataLoader
from hamu_tool.utils import CorpusReader
from tqdm import tqdm

dataset = 'nfcorpus'

loader = DataLoader.load(f'beir/{dataset}')
reader_q = CorpusReader(f'data/{dataset}/query.gpt35.base.idx')

vocab = set()
for doc in tqdm(loader.get_docs(), desc=f'[{dataset}] Building vocab'):
    if hasattr(doc, 'title'):
        words = f'{doc.title} {doc.text}'.lower().split()
    else:
        words = doc.text.lower().split()
    vocab.update(words)

# oov_ratio_list = []
# for query in tqdm(reader_q, desc=f'[{dataset}] Calculating OOV ratio'):
#     words = query.text.lower().split()
#     oov_cnt = 0
#     for word in words:
#         if not word in vocab:
#             oov_cnt += 1
#     oov_ratio = 100 * oov_cnt / len(words)
#     oov_ratio_list.append(oov_ratio)
# oov_ratio = sum(oov_ratio_list) / len(oov_ratio_list)
# print(f'OOV ratio: {oov_ratio:.2f}%')

oov_cnt = 0
word_cnt = 0
with open(f'data/{dataset}/keywords.txt', 'r') as file:
    for line in tqdm(file, desc=f'[{dataset}] Calculating OOV ratio'):
        words = line.strip().lower().split()
        for word in words:
            if not word in vocab:
                oov_cnt += 1
            word_cnt += 1
oov_ratio = 100 * oov_cnt / word_cnt
print(f'OOV ratio: {oov_ratio:.2f}%')
