import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import time
import ujson
import torch
import pickle
import copy
import random
random.seed(42)

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from argparse import ArgumentParser
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from IsoScore import IsoScore, existing_scores

from colbert.modeling.inference import ModelInference

from utils import *

def avg_cos(embs):
    n = len(embs)
    embs = embs @ embs.T
    return (torch.sum(embs) - n) / (n * (n-1))

def late_interaction(Q, D):
    if Q.dtype != D.dtype:
        Q = Q.to(D.dtype)
    score = Q @ D.T
    return score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=None, required=True, type=str)
    arg_ = parser.parse_args()
    model_path = arg_.model_path
    
    checkpoint = f"/data/colbert-prf/experiments/{model_path}/train.py/msmarco.psg.cosine/checkpoints/colbert-200000.dnn"
    model, checkpoint = load_model(checkpoint=checkpoint, do_print=True)
    model = ModelInference(model, amp=True)

    dataset_list = ["msmarco"]
    scores = {"Avg-Cos":{i:[] for i in range(13)}, "1-IsoScore":{i:[] for i in range(13)}, "1-Partition":{i:[] for i in range(13)}}


    if not os.path.isfile("metric/metric_compared_pids.pickle"):
        pid_to_doc = {}
        for dataset in tqdm(dataset_list):
            pid_to_doc[dataset] = read_corpus(f"/data/beir/{dataset}/corpus.jsonl")
            pids = random.sample(list(pid_to_doc[dataset]),k=10_000)
            pid_to_doc[dataset] = {pid:pid_to_doc[dataset][pid] for pid in pids}
        pickle_dump(pid_to_doc, "metric_compared_pids.pickle")
    else:
        print("Load pid_to_doc.")
        pid_to_doc = pickle_load("metric/metric_compared_pids.pickle")


    for dataset in tqdm(dataset_list):
        passage_list = list(pid_to_doc[dataset].values())[:100]

        D_ids, _ = model.doc_tokenizer.tensorize(passage_list, bsize=512)
        masks = torch.cat([masks for input_ids,masks in D_ids]).cuda()
        D_ids = torch.cat([input_ids for input_ids,masks in D_ids]).cuda()

        with torch.no_grad():
            D = model.colbert.bert(D_ids, attention_mask=masks, output_hidden_states=True)
            for ith_layer, hidden_states in tqdm(enumerate(D.hidden_states)):
                for embs in tqdm(hidden_states):
                    # Avg-Cos
                    score = avg_cos(embs).cpu().to(torch.float16).item()
                    scores["Avg-Cos"][ith_layer].append(score)
                        
                    # IsoScore
                    score = 1 - IsoScore.IsoScore(embs.T).cpu().to(torch.float16).item()
                    scores["1-IsoScore"][ith_layer].append(score)
                        
                    # Partition
                    score = 1 - existing_scores.partition_score(embs.T.cpu().numpy())#.cpu().to(torch.float16).item()
                    scores["1-Partition"][ith_layer].append(score)
    
    pickle_dump(scores, f"metric/metric_layer_{model_path}_1k.pickle")
    print("Save done.")