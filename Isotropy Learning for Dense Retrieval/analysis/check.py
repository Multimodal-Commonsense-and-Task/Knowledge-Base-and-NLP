import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import time
import ujson
import torch
import pickle
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from argparse import ArgumentParser

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from colbert.modeling.inference import ModelInference

from utils import *

if __name__ == "__main__":
    dataset_list = ["trec-covid", "bioasq", "nfcorpus", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news", \
                    "robust04", "arguana", "touche", "quora", "dbpedia-entity", "scidocs", "fever", \
                    "climate-fever", "scifact", "cqadupstack/android", "cqadupstack/english", \
                    "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", "cqadupstack/physics", \
                    "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", \
                    "cqadupstack/webmasters", "cqadupstack/wordpress"]
    hypo_colbert = pickle_load(f"/data/colbert-prf/analysis/hypo/hypo-sensim-single-lambda-1-bm25/hypo_colbert_bm25.pickle")
    colbert_sentropy = pickle_load(f"/data/colbert-prf/analysis/hypo/hypo-sensim-single-lambda-1-bm25/colbert_sentropy_rmsptok.pickle")
    print(f"hypo_colbert keys: {hypo_colbert.keys()}")
    print(f"colbert_sentropy keys: {colbert_sentropy.keys()}")
    sensim = [score for dataset in dataset_list for qid, score in colbert_sentropy[dataset].items()]
    print(f"sensim: {np.mean(sensim)}")