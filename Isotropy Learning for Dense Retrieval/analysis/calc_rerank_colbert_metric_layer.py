import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
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
    scores = {"Avg-Cos":[], "1-IsoScore":[], "1-Partition":[]}


    vocab_size = model.colbert.bert.config.vocab_size
    index = [i for i in range(vocab_size)]

    for dataset in tqdm(dataset_list):
        #D_ids = random.sample(index, k=10_000)
        D_ids = index
        D_ids = torch.tensor(D_ids).cuda()
        D_ids = D_ids.unsqueeze(dim=-1)
        masks = torch.ones(D_ids.shape).cuda()

        with torch.no_grad():
            D = model.colbert.bert(D_ids, attention_mask=masks)
            embs = D.embedding_output
            embs = embs.squeeze()

            #Avg-Cos
            score = avg_cos(embs).cpu().to(torch.float16).item()
            scores["Avg-Cos"] = score
            print("Avg-Cos done.")

            #IsoScore
            score = 1 - IsoScore.IsoScore(embs.T).cpu().to(torch.float16).item()
            scores["1-IsoScore"] = score
            print("1-IsoScore done.")

            # Partition
            score = 1 - existing_scores.partition_score(embs.T.cpu().numpy())#.cpu().to(torch.float16).item()
            scores["1-Partition"] = score
            print("1-Partition done.")
    
    pickle_dump(scores, f"metric/metric_vocab_layer_{model_path}_1k.pickle")
    print("Save done.")