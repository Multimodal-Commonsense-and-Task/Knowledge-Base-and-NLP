import os
import time
import pickle
import numpy as np
import pandas as pd

from utils import *


if __name__ == "__main__":
    dataset_list = ["msmarco", "trec-covid", "bioasq", "nfcorpus", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news", \
                    "robust04", "arguana", "touche", "quora", "dbpedia-entity", "scidocs", "fever", \
                    "climate-fever", "scifact", "cqadupstack/android", "cqadupstack/english", \
                    "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", "cqadupstack/physics", \
                    "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", \
                    "cqadupstack/webmasters", "cqadupstack/wordpress"]


    ret = {}
    with open("trial_rerank_colbertv2_HIL_100k_len300_bm25.txt", "r") as f:
        for line in f:
            line = line.strip()

            if "ndcg@10:" in line:
                print(line.split())
                dataset = line.split()[1]
                ndcg = line.split()[-1]
                ret[dataset] = ndcg
    


    for dataset in dataset_list:
        if dataset not in ret:
            print(f"{dataset} not in ret!")
        else:
            print(ret[dataset])
