import argparse
import faiss
import json
import numpy as np
import os

from alive_progress import alive_bar
from datasets import load_dataset, Dataset
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

QUERY_TYPE_MAPPING = {
    'original': 'examples',
    'gpt4': 'gpt4_reason',
    'llama3': 'llama3-70b_reason',
    'claude3': 'claude-3-opus_reason',
    'gemini1': 'Gemini-1.0_reason',
    'grit': 'grit_reason',
}

def get_total_count(path : str):
    total = 0
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            total += 1
    return total

def get_dataset(name : str):
    os.makedirs(f'data/{name}', exist_ok=True)

    for query_type in QUERY_TYPE_MAPPING.keys():
        if os.path.exists(f'data/{name}/query.{query_type}.jsonl'):
            continue
        dataset = load_dataset('xlangai/BRIGHT', QUERY_TYPE_MAPPING[query_type])[name]
        assert isinstance(dataset, Dataset)
        with open(f'data/{name}/query.{query_type}.jsonl', 'w', encoding='utf-8') as fp, \
            alive_bar(total=len(dataset), title=f'[{name}] Download Queries ({query_type})') as bar:
            for data in dataset:
                assert isinstance(data, dict)
                dump = json.dumps({
                    'qid': data['id'],
                    'text': data['query'],
                    'pos_dids': data['gold_ids'],
                    'pos_dids_long': data['gold_ids_long'],
                    'excluded_dids': data['excluded_ids'],
                    'gold_answer': data['gold_answer'],
                    'reasoning': data['reasoning'],
                })
                fp.write(f'{dump}\n')
                bar()

def build_bm25_index(name : str):
    os.makedirs(f'index/{name}/bm25', exist_ok=True)
    indexer = LuceneIndexer(f'index/{name}/bm25', args=['-index', f'index/{name}/bm25', '-storeRaw'])
    dataset = load_dataset('xlangai/BRIGHT', 'documents')[name]
    assert isinstance(dataset, Dataset)
    with alive_bar(total=len(dataset), title=f'[{name}] Sparse Indexing Documents') as bar:
        for data in dataset:
            assert isinstance(data, dict)
            indexer.add_doc_dict({'id': data['id'], 'contents': data['content']})
            bar()
    indexer.close()

def build_reasonir_index(name : str, batch_size : int):
    os.makedirs(f'index/{name}/reasonir', exist_ok=True)
    model = SentenceTransformer('reasonir/ReasonIR-8B', trust_remote_code=True, model_kwargs={'torch_dtype': 'auto'})
    searcher = LuceneSearcher(f'index/{name}/bm25')
    index = None
    batch_len = (searcher.num_docs + batch_size - 1) // batch_size
    with open(f'index/{name}/reasonir/docid', 'w') as fp_id, \
        alive_bar(total=batch_len, title=f'[{name}] Dense Indexing Documents') as bar:
        for i in range(0, searcher.num_docs, batch_size):
            docs = []
            for j in range(i, min(i + batch_size, searcher.num_docs)):
                doc = searcher.doc(j)
                assert doc is not None
                doc_raw = doc.raw() if doc else ''
                docs.append(json.loads(doc_raw)['contents'].strip())
                fp_id.write(f'{doc.docid()}\n')
            embs = model.encode(docs, instruction='<|embed|>\n')
            if index is None:
                index = faiss.IndexFlatIP(embs.shape[1])
            index.add(np.ascontiguousarray(embs)) # type: ignore
            bar()
    faiss.write_index(index, f'index/{name}/reasonir/index')

def main(args):
    get_dataset(args.dataset)
    build_bm25_index(args.dataset)
    build_reasonir_index(name=args.dataset, batch_size=args.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess BRIGHT dataset')
    parser.add_argument('--dataset', type=str, default='biology', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    args = parser.parse_args()

    main(args)
