import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import ujson
import pickle
import copy
import pytrec_eval
import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser

import faiss
assert faiss.get_num_gpus() > 0

import pyterrier as pt
pt.init()

from analysis.utils import *

def get_hypothesis(model, dataset, queries):
    print(f"len(qrels): {len(qrels)}")
    print(f"before len(queries): {len(queries)}")
    queries = {_id: q_text for _id, q_text in queries.items() if _id in qrels}
    print(f"after len(queries): {len(queries)}")
    assert len(qrels) == len(queries), f"qrels: {len(qrels)}, queries: {len(queries)}"

    input_df = pd.DataFrame(queries.items(), columns=["qid", "query"])
    df = model(input_df)[["qid","docid","islx","iscb"]]
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--beir_dataset", default=None, required=True, type=str)
    parser.add_argument("--model_path", default=None, required=True, type=str)
    parser.add_argument("--model_step", default=None, required=False, type=str)
    parser.add_argument("--index_path", default=None, required=True, type=str)
    parser.add_argument("--qidf", default=None, required=False, type=str)

    arg_ = parser.parse_args()
    dataset = arg_.beir_dataset
    model_path = arg_.model_path
    model_step = arg_.model_step
    index_path = arg_.index_path
    qidf = arg_.qidf
    
    file_name = dataset.replace("/","-")

    #_id_to_qid = pickle_load(f"/data/beir/{dataset}/_id_to_qid.pickle")
    cbpid_to_id = pickle_load(f"{index_path}/{dataset}/cbpid_to_id.pickle")
    _id_to_cbpid = pickle_load(f"{index_path}/{dataset}/_id_to_cbpid.pickle")

    if dataset == "msmarco":
        qrels = read_qrels(f"/data/beir/{dataset}/qrels/dev.tsv")
    else:
        qrels = read_qrels(f"/data/beir/{dataset}/qrels/test.tsv")
    queries = read_queries(f"/data/beir/{dataset}/queries.jsonl")
    cbpid_to_doc = {_id_to_cbpid[_id]:doc for _id, doc in read_corpus(f"/data/beir/{dataset}/corpus.jsonl").items()}

    from pyterrier_colbert.ranking import ColBERTFactory
    index=(f"{index_path}", f"{dataset}")

    if model_step:
        model_path = f"experiments/{model_path}/train.py/msmarco.psg.cosine/checkpoints/colbert-{model_step}.dnn"
    else:
        model_path = f"/data/ColBERT/experiments/checkpoint/{model_path}/pytorch_model.bin"
    pytcolbert = ColBERTFactory(model_path, *index, cbpid_to_doc=cbpid_to_doc, is_ours=(qidf is not None), qidf=qidf)
    dense_e2e = pytcolbert.first_stage()
    first_stage = get_hypothesis(dense_e2e, dataset, queries)
    
    first_stage.to_csv(f"{index_path}/{dataset}/first_stage.tsv", sep="\t", index=False, header=False)
    print(f"dataset: {dataset} done.")