import pandas as pd
import pickle
import os

from tqdm import tqdm

import faiss
assert faiss.get_num_gpus() > 0

import pyterrier as pt
pt.init()


def pickle_dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def read_qrels(path):
    qrels = {}
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            qid, pos, *other = line.split("\t")
            assert len(other) == 0, line
            
            qid = int(qid)
            pos = int(pos)
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(pos)
    return qrels

def read_collection(path):
    collection = {}
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            pid, title, *other = line.split("\t")
            assert len(other) == 0, line
            
            pid = int(pid)
            assert pid not in collection, pid
            collection[pid] = title
    return collection

def make_hypothesis(step, path, model):
    model_name = "-".join(path.split("-")[:2])
    print(model_name)
    os.makedirs(os.path.join("hypothesis-nonpsg", step), exist_ok=True)
    path = os.path.join("hypothesis-nonpsg", step, path)

    qids = list(set(qid_to_esci_pid.keys()))
    with open(path, "w") as f:
        for i, qid in enumerate(qids):
            if i % 1000 == 0:
                print(f"{i}/{len(qids)}, {round(100*i/len(qids),2)}%")
            query_id = qid_to_queryid[qid]
            query = qid_to_query[qid]

            df = model.search(query)[["query", "docid", "score", "rank"]]
            df["product_id"] = df["docid"].apply(lambda x: pid_to_productid[x])
            assert len(df) == len(qid_to_candidate_pid[qid]), f"{len(df)}, {len(qid_to_candidate_pid[qid])}"

            for product_id, rank, score,  in zip(df["product_id"], df["rank"], df["score"]):
                f.write(f"{query_id} Q0 {product_id} {rank} {round(score,3)} {model_name}\n")


dataset = pt.get_dataset("trec-deep-learning-passages")
beta = 0.5
checkpoint = "models/colbert.dnn"

from pyterrier_colbert.ranking import ColBERTFactory
index=("index",f"trec")

pytcolbert = ColBERTFactory(checkpoint, *index)
dense_e2e = pytcolbert.end_to_end()
prf_rank = pytcolbert.prf(rerank=False, beta=beta)
prf_rerank = pytcolbert.prf(rerank=True, beta=beta)


from pyterrier.measures import *
pt.Experiment(
    [
        dense_e2e,
        prf_rank,
        prf_rerank
    ],
    dataset.get_topics('test-2019'),
    dataset.get_qrels('test-2019'),
    eval_metrics=[ AP(rel=2)@1000, nDCG@10, RR(rel=2)@10, "mrt"],
    batch_size=10,
    drop_unused=True,
    names = ["ColBERT E2E","ColBERT-PRF Ranker beta=1","ColBERT-PRF ReRanker beta=1"]
)


#dense_e2e = pytcolbert.end_to_end()


"""
esci_path = "../esci-data/shopping_queries_dataset"
qid_to_candidate_pid = pickle_load(os.path.join(esci_path, "qid_to_candidate_pid.pickle"))
qid_to_esci_pid = pickle_load(os.path.join(esci_path, "qid_to_esci_pid.pickle"))
pid_to_productid = pickle_load(os.path.join(esci_path, "pid_to_productid.pickle"))
qid_to_queryid = pickle_load(os.path.join(esci_path, "qid_to_queryid.pickle"))
qid_to_query = pickle_load(os.path.join(esci_path, "qid_to_query.pickle"))
query_to_qid = pickle_load(os.path.join(esci_path, "query_to_qid.pickle"))

qrels_test = read_qrels(os.path.join(esci_path, "qrels_test.tsv"))
collection = read_collection(os.path.join(esci_path, "collection.tsv"))


make_hypothesis(f"{step}", "colbert-hypothesis.results", dense_e2e)
make_hypothesis(f"{step}", f"colbert-rank-beta{beta}-hypothesis.results", prf_rank)
make_hypothesis(f"{step}", f"colbert-rerank-beta{beta}-hypothesis.results", prf_rerank)
"""
