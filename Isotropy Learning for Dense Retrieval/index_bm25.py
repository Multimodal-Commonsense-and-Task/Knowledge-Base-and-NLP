import os
import ujson
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

import argparse
from argparse import ArgumentParser

import pyterrier as pt
pt.init()
from pyterrier.datasets import Dataset as dataset
from pyterrier_colbert.indexing import ColBERTIndexer

from analysis.utils import *


def read_corpus(path, valid_pids=None):
    iterator = []
    pid_idx = 0
    cbpid_to_id = {}
    _id_to_cbpid = {}

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            data = ujson.loads(line)
            if "_id" not in data:
                _id = data["id"]
            else:
                _id = data["_id"]
            if valid_pids is not None and _id not in valid_pids:
                continue
            title = data["title"].strip()
            text = data["text"].strip()
            if len(title) == 0 and len(text) == 0:
                continue
            iterator.append({"docno": pid_idx, "text": " ".join([title, text])})
            cbpid_to_id[pid_idx] = _id
            _id_to_cbpid[_id] = pid_idx
            pid_idx += 1

    return iterator, cbpid_to_id, _id_to_cbpid


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--beir_dataset", default=None, required=True, type=str)
    parser.add_argument("--index_path", default=None, required=True, type=str)
    parser.add_argument("--model_path", default=None, required=True, type=str)
    parser.add_argument("--qidf", default=None, required=False, type=str)
    parser.add_argument('--local_rank', dest='rank', default=-1, type=int)
    arg_ = parser.parse_args()
    
    dataset = arg_.beir_dataset
    rank = arg_.rank
    index_path = arg_.index_path
    model_path = arg_.model_path
    qidf = arg_.qidf
    file_name = dataset.replace("/","-")

    qid_to_bm25 = read_bm25_results(f"/data/bm25/run.beir-bm25-flat.{file_name}.txt")
    valid_pids = set(flatten_2d([pids for qid, pids in qid_to_bm25.items()]))
    print(f"Start {dataset} indexing.")
    print(f"model_path path: {model_path}")
    print(f"index path: {index_path}")
    print(f"# of pids: {len(valid_pids)}")

    indexer = ColBERTIndexer(f"experiments/{model_path}/train.py/msmarco.psg.cosine/checkpoints/colbert-200000.dnn", index_path, f"{dataset}", chunksize=6, rank=rank, qidf=qidf)
    corpus_iter, cbpid_to_id, _id_to_cbpid = read_corpus(f"/data/beir/{dataset}/corpus.jsonl", valid_pids=valid_pids)

    os.makedirs(f"{index_path}/{dataset}", exist_ok=True)
    pickle_dump(cbpid_to_id, f"{index_path}/{dataset}/cbpid_to_id.pickle")
    pickle_dump(_id_to_cbpid, f"{index_path}/{dataset}/_id_to_cbpid.pickle")

    indexer.index(corpus_iter)
    print(f"Done {dataset} indexing.")