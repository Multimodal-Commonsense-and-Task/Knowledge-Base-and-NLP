import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import time
import ujson
import torch
import pickle
import copy
import random
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from argparse import ArgumentParser

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from colbert.modeling.inference import ModelInference

from utils import *


def late_interaction(Q, D):
    if Q.dtype != D.dtype:
        Q = Q.to(D.dtype)
    #maxscoreQ, argmaxscoreQ = (Q @ D.permute(0, 2, 1)).max(2)
    score = Q @ D.permute(0, 2, 1)
    return score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--beir_dataset", default=None, required=True, type=str)
    parser.add_argument("--model_path", default=None, required=True, type=str)
    parser.add_argument("--model_step", default=None, required=False, type=str)
    parser.add_argument("--index_path", default=None, required=False, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--bm25", default=False, action='store_true')
    arg_ = parser.parse_args()
    dataset_list = [arg_.beir_dataset]
    model_path = arg_.model_path
    model_step = arg_.model_step
    index_path = arg_.index_path
    save_path = arg_.save_path
    bm25 = arg_.bm25
    os.makedirs(save_path, exist_ok=True)
    random.seed(42)
    
    if bm25:
        bm25_filename = "_bm25"
    else:
        bm25_filename = ""
    if os.path.isfile(f"{save_path}/hypo_sensim_lx{bm25_filename}.pickle"):
        hypo_lx = pickle_load(f"{save_path}/hypo_sensim_lx{bm25_filename}.pickle")
        hypo_cb = pickle_load(f"{save_path}/hypo_sensim_cb{bm25_filename}.pickle")
        hypo_ours = pickle_load(f"{save_path}/hypo_sensim_ours{bm25_filename}.pickle")
        # hypo_lx = {}
        # hypo_cb = {}
        # hypo_ours = {}

        # lx_sensim = pickle_load(f"{save_path}/lx_score_dist.pickle")
        # cb_sensim = pickle_load(f"{save_path}/cb_score_dist.pickle")
    else:
        hypo_lx = {}
        hypo_cb = {}
        hypo_ours = {}

        # lx_sensim = {}
        # cb_sensim = {}

    # short_dataset_list = ["trec-covid", "bioasq", "nfcorpus", "fiqa", "signal1m", "trec-news", "robust04", "arguana", "touche", "dbpedia-entity", "scidocs", "scifact",
    #                     "cqadupstack/android", "cqadupstack/english", "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", \
    #                     "cqadupstack/physics", "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", \
    #                     "cqadupstack/webmasters", "cqadupstack/wordpress"]

    # short_dataset_list = [dataset for dataset in short_dataset_list if os.path.isfile(f"/data/colbert-prf/index/{index_path}/{dataset}/first_stage.tsv")]
    # short_dataset_list = [dataset for dataset in short_dataset_list if dataset not in dataset_list]
    # dataset_list = short_dataset_list + dataset_list
    # dataset_list = [dataset for dataset in dataset_list if dataset not in hypo_ours.keys()]

    print(f"Start {dataset_list} reranking.")
    print(f"model path: {model_path}")

    qidf = {}
    for dataset in dataset_list:
        qidf[dataset] = pickle_load(f"/data/beir/{dataset}/qidf.pickle")

    if model_step:
        model_path = f"/data/colbert-prf/experiments/{model_path}/train.py/msmarco.psg.cosine/checkpoints/colbert-{model_step}.dnn"
    else:
        model_path = f"/data/ColBERT/experiments/checkpoint/{model_path}/pytorch_model.bin"
    model_ours, checkpoint = load_model(checkpoint=model_path, qidf=qidf, do_print=True)
    inference_ours = ModelInference(model_ours, amp=True)
    for idx, dataset in tqdm(enumerate(dataset_list)):
        # Load dataset
        qrels = {}
        queries = {}
        pid_to_doc = {}
        qid_to_pids = {}
        file_name = dataset.replace("/","-")

        if "msmarco" == dataset:
            qrels[dataset] = read_qrels(f"/data/beir/{dataset}/qrels/dev.tsv")
        else:
            qrels[dataset] = read_qrels(f"/data/beir/{dataset}/qrels/test.tsv")
        queries[dataset] = read_queries(f"/data/beir/{dataset}/queries.jsonl", qrels[dataset])
        if bm25:
            qid_to_pids[dataset] = read_bm25_results(f"/data/bm25/run.beir-bm25-flat.{file_name}.txt")
        else:
            cbpid_to_id = pickle_load(f"/data/colbert-prf/index/{index_path}/{dataset}/cbpid_to_id.pickle")
            _, _, qid_to_pids[dataset] = read_first_stage(f"/data/colbert-prf/index/{index_path}/{dataset}/first_stage.tsv", cbpid_to_id=cbpid_to_id)
        valid_pids = set(flatten_2d([pids for qid, pids in qid_to_pids[dataset].items()]))
        pid_to_doc[dataset] = read_corpus(f"/data/beir/{dataset}/corpus.jsonl", valid_pids=valid_pids)
        print(f"{dataset} load done.")


        # Initialize
        qid_to_embs, qid_to_lx_embs, qid_to_ids, qid_to_masks = {}, {}, {}, {}
        pid_to_embs, pid_to_lx_embs, pid_to_ids = {}, {}, {}

        lx_scores_list, cb_scores_list = [], []

        hypo_lx[dataset] = {}
        hypo_cb[dataset] = {}
        hypo_ours[dataset] = {}

        # lx_sensim[dataset] = {}
        # cb_sensim[dataset] = {}

        # qids encoding
        start = time.time()
        qids = list(set([qid for qid, _ in qrels[dataset].items()]))
        print(f"\n{dataset}: {len(qids)} qids encoding...")
        texts = [queries[dataset][qid] for qid in qids]
        Q, lx_Q, Q_ids, masks = inference_ours.queryFromText(texts, bsize=512, with_ids=True)
        Q, lx_Q, Q_ids, masks = Q.detach().cpu(), lx_Q.detach().cpu(), Q_ids.cpu(), masks.cpu()
        for qid, q, lx_q, q_ids, q_mask in zip(qids, Q, lx_Q, Q_ids, masks):
            qid_to_embs[qid] = q
            qid_to_lx_embs[qid] = lx_q
            qid_to_ids[qid] = q_ids
            qid_to_masks[qid] = q_mask
        print(f"{dataset}: {len(qids)} qids Done!")
        print(f"qids encoding time: {time.time()-start}s.\n")

        # pids encoding
        start = time.time()
        pids = list(set(flatten_2d([pids for qid, pids in qid_to_pids[dataset].items()])))
        print(f"{dataset}: {len(pids)} pids encoding...")
        docs = [pid_to_doc[dataset][pid] for pid in pids]
        pids_batches = split_into_batches(pids, bsize=16384)
        docs_batches = split_into_batches(docs, bsize=16384)

        for i, docs_batch in enumerate(docs_batches):
            print(f"{dataset}: {i}/{len(docs_batches)}\t{round(100*i/len(docs_batches),2)}%\t{round(time.time()-start,2)}s.")
            D, lx_D, D_ids = inference_ours.docFromText(docs_batch, bsize=512, keep_dims=False, with_ids=True)
            pids_batch = pids_batches[i]
            for pid, d, lx_d, d_ids in zip(pids_batch, D, lx_D, D_ids):
                pid_to_embs[pid] = d
                pid_to_lx_embs[pid] = lx_d
                pid_to_ids[pid] = d_ids
        print(f"{dataset}: {len(pids)} pids Done!")
        print(f"pids encoding time: {time.time()-start}s.\n")

        # lx_score_dist, cb_score_dist = [], []
        # compute late-interaction
        for qid, pids in tqdm(qid_to_pids[dataset].items()):
            Q, lx_Q, Q_ids, masks = qid_to_embs[qid], qid_to_lx_embs[qid], qid_to_ids[qid], qid_to_masks[qid]
            Q, lx_Q, Q_ids, masks = Q.cuda(), lx_Q.cuda(), Q_ids.cuda(), masks.cuda()
            q_weights = torch.tensor([qidf[dataset].get(qtok.item(),0) for qtok in Q_ids]).cuda()
            q_len = torch.sum(masks).item()

            # compute hypothesis
            lx_tmp, cb_tmp = [], []
            #pids_batches = split_into_batches(pids, bsize=1024)
            all_pids_batches = split_into_batches(pids, bsize=16384)

            for pids_batches in all_pids_batches:
                D = [pid_to_embs[pid] for pid in pids_batches]
                lx_D = [pid_to_lx_embs[pid] for pid in pids_batches]
                D = pad_sequence(D, batch_first=True).cuda()
                lx_D = pad_sequence(lx_D, batch_first=True).cuda()

                D_batches = split_into_batches(D, bsize=512)
                lx_D_batches = split_into_batches(lx_D, bsize=512)

                #for batch in pids_batches:
                for D, lx_D in zip(D_batches, lx_D_batches):
                    # D = [pid_to_embs[pid] for pid in batch]
                    # lx_D = [pid_to_lx_embs[pid] for pid in batch]
                    # D = pad_sequence(D, batch_first=True).cuda()
                    # lx_D = pad_sequence(lx_D, batch_first=True).cuda()

                    # compute hypothesis
                    dot = late_interaction(Q, D)
                    maxscoreQ, argmaxscoreQ = dot.max(2)

                    lx_dot = late_interaction(lx_Q, lx_D)
                    lx_maxscoreQ, lx_argmaxscoreQ = lx_dot.max(2)
                    lx_maxscoreQ = q_weights * lx_maxscoreQ

                    lx_tmp.extend(lx_maxscoreQ.cpu().tolist())
                    cb_tmp.extend(maxscoreQ.cpu().tolist())


                    # compute score dist
                    # if dataset == "msmarco":
                    #     D_ids = [pid_to_ids[pid] for pid in batch]
                    #     D_ids = pad_sequence(D_ids, batch_first=True, padding_value=-100).cuda()
                    #     valid_pid_toks = torch.sum(D_ids != -100, dim=-1)
                    #     lx_scores = lx_dot[:, :q_len]
                    #     cb_scores = dot[:, :q_len]

                    #     for cb_score, lx_score, d_len in zip(cb_scores, lx_scores, valid_pid_toks):
                    #         lx_score_dist.append(lx_score[:,:d_len].reshape(-1).to(torch.float16).cpu())
                    #         cb_score_dist.append(cb_score[:,:d_len].reshape(-1).to(torch.float16).cpu())

            lx_scores_list.append(lx_tmp)
            cb_scores_list.append(cb_tmp)
        
        # if dataset == "msmarco":
        #     lx_score_dist = torch.cat(lx_score_dist)
        #     lx_score_dist = lx_score_dist[random.sample(range(len(lx_score_dist)), k=10_000)]
        #     cb_score_dist = torch.cat(cb_score_dist)
        #     cb_score_dist = cb_score_dist[random.sample(range(len(cb_score_dist)), k=10_000)]
        #     lx_sensim[dataset] = lx_score_dist
        #     cb_sensim[dataset] = cb_score_dist

        if False and dataset in ["arguana", "hotpotqa", "fever", "climate-fever", "msmarco"]:
            flattened_lx_scores_list = random.sample(flatten_3d(random.sample(lx_scores_list, k=int(len(lx_scores_list)/5))),k=1000000)
            flattened_cb_scores_list = random.sample(flatten_3d(random.sample(cb_scores_list, k=int(len(cb_scores_list)/5))),k=1000000)
        else:
            # flattened_lx_scores_list = random.sample(flatten_3d(lx_scores_list),k=1000000)
            # flattened_cb_scores_list = random.sample(flatten_3d(cb_scores_list),k=1000000)
            flattened_lx_scores_list = flatten_3d(lx_scores_list)
            flattened_cb_scores_list = flatten_3d(cb_scores_list)

        m_lx, std_lx = np.mean(flattened_lx_scores_list), np.std(flattened_lx_scores_list)
        m_cb, std_cb = np.mean(flattened_cb_scores_list), np.std(flattened_cb_scores_list)
        
        assert len(qid_to_pids[dataset]) == len(lx_scores_list) == len(cb_scores_list)
        for (qid, pids), lx_scores, cb_scores in zip(tqdm(qid_to_pids[dataset].items()), lx_scores_list, cb_scores_list):
            hypo_lx[dataset][qid] = {}
            hypo_cb[dataset][qid] = {}
            hypo_ours[dataset][qid] = {}

            assert len(lx_scores) == len(cb_scores) == len(pids), f"{len(lx_scores)}, {len(cb_scores)}, {len(pids)}"
            for pid, lx_score, cb_score in zip(pids, lx_scores, cb_scores):
                lx_score = (lx_score - m_lx) / std_lx
                cb_score = (cb_score - m_cb) / std_cb
                lx_score = lx_score.sum(0)
                cb_score = cb_score.sum(0)

                hypo_lx[dataset][qid][pid] = lx_score
                hypo_cb[dataset][qid][pid] = cb_score
                hypo_ours[dataset][qid][pid] = (lx_score + cb_score) / 2

        lx_ndcg, lx_mrr = evaluation(hypo_lx[dataset], qrels[dataset])
        cb_ndcg, cb_mrr = evaluation(hypo_cb[dataset], qrels[dataset])
        ours_ndcg, ours_mrr = evaluation(hypo_ours[dataset], qrels[dataset])
        print(f"{model_path} {dataset} ndcg@10: {lx_ndcg}\t{cb_ndcg}\t{ours_ndcg}")
        print(f"{model_path} {dataset} mrr@10: {lx_mrr}\t{cb_mrr}\t{ours_mrr}")

        if (idx%10 == 9) or (idx == len(dataset_list)-1):
            pass
            # pickle_dump(hypo_lx, f"{save_path}/hypo_sensim_lx{bm25_filename}.pickle")
            # pickle_dump(hypo_cb, f"{save_path}/hypo_sensim_cb{bm25_filename}.pickle")
            # pickle_dump(hypo_ours, f"{save_path}/hypo_sensim_ours{bm25_filename}.pickle")
            # print(f"Saving {model_path} {dataset}, {len(hypo_ours)} are saved!")
        
            # if dataset == "msmarco":
            #    pickle_dump(lx_sensim, f"{save_path}/lx_score_dist.pickle")
            #    pickle_dump(cb_sensim, f"{save_path}/cb_score_dist.pickle")


        # if dataset == "msmarco":
        #     lx_sensim_score = torch.mean(lx_sensim[dataset]).item()
        #     cb_sensim_score = torch.mean(cb_sensim[dataset]).item()
        #     print(f"{model_path} {dataset} lx_sensim_score: {round(lx_sensim_score,6)}")
        #     print(f"{model_path} {dataset} cb_sensim_score: {round(cb_sensim_score,6)}")
        print(f"{model_path} {dataset} done.")