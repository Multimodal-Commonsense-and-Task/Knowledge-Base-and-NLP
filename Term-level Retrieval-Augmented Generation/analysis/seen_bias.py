import json
import os
from hamu_tool.dataset import DataLoader
from hamu_tool.utils import CorpusReader

def seen_bias(dataset):
    loader = DataLoader.load(f'beir/{dataset}')

    if not os.path.exists(f'data/{dataset}/query.gpl.idx'):
        CorpusReader.build_index(f'data/{dataset}/query.gpl.jsonl', f'data/{dataset}/query.gpl.idx')
    reader_q = CorpusReader(f'data/{dataset}/query.gpl.idx')

    seen_ratios = []
    unseen_ratios = []
    with open(f'data/{dataset}/qrel.gpl.tsv', 'r', encoding='utf-8') as fp:
        for line in fp:
            gen_qid, _, did, _ = line.strip().split('\t')
            gen_query = reader_q[gen_qid]
            doc = loader.get_doc(did)
            qid = loader.get_drel('test', did)[0].qid
            query = loader.get_query(qid)

            query_term = set(query.text.split())
            doc_term = set(doc.text.split())
            gen_query_term = set(gen_query.text.split())

            seen = query_term & doc_term
            unseen = query_term - doc_term

            if len(seen) * len(unseen) == 0:
                continue
            seen_recall = len(gen_query_term & seen) / len(seen)
            unseen_recall = len(gen_query_term & unseen) / len(unseen)

            if seen_recall + unseen_recall < 1e-6:
                continue
            seen_ratio = seen_recall / (seen_recall + unseen_recall)
            unseen_ratio = unseen_recall / (seen_recall + unseen_recall)

            seen_ratios.append(seen_ratio)
            unseen_ratios.append(unseen_ratio)

    seen_ratio = 100 * sum(seen_ratios) / len(seen_ratios)
    unseen_ratio = 100 * sum(unseen_ratios) / len(unseen_ratios)
    print(f'{dataset}: {seen_ratio:.2f} {unseen_ratio:.2f}')

dataset_list = ['nfcorpus', 'scifact', 'scidocs', 'fiqa']
# dataset_list = ['arguana', 'bioasq', 'climate-fever', 'dbpedia', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'robust04', 'scidocs', 'scifact', 'signal1m', 'touche-v2', 'trec-covid', 'trec-news']
for dataset in dataset_list:
    seen_bias(dataset)
