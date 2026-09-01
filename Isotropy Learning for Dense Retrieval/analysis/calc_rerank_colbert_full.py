import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
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
import torch.nn.functional as F

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
    parser.add_argument("--post_processing", default=None, type=str)
    arg_ = parser.parse_args()
    dataset_list = [arg_.beir_dataset]
    model_path = arg_.model_path
    model_step = arg_.model_step
    index_path = arg_.index_path
    save_path = arg_.save_path
    bm25 = arg_.bm25
    post_processing = arg_.post_processing
    if post_processing:
        assert post_processing in ["whitening", "glow", "nice"]
    os.makedirs(save_path, exist_ok=True)
    random.seed(42)


    if bm25:
        bm25_filename = "_bm25"
    else:
        bm25_filename = ""
    if os.path.isfile(f"{save_path}/hypo_colbert{bm25_filename}.pickle"):
        hypo_colbert = pickle_load(f"{save_path}/hypo_colbert{bm25_filename}.pickle")
        #colbert_sensim = pickle_load(f"{save_path}/colbert_score_dist.pickle")
    else:
        hypo_colbert = {}
        #colbert_sensim = {}

    # short_dataset_list = ["trec-covid", "bioasq", "nfcorpus", "fiqa", "signal1m", "trec-news", "robust04", "arguana", "touche", "dbpedia-entity", "scidocs", "scifact",
    #                     "cqadupstack/android", "cqadupstack/english", "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", \
    #                     "cqadupstack/physics", "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", \
    #                     "cqadupstack/webmasters", "cqadupstack/wordpress"]
    # short_dataset_list = ["robust04", "trec-news", "trec-covid", "nq"]
    # short_dataset_list = [dataset for dataset in short_dataset_list if os.path.isfile(f"/data/colbert-prf/index/{index_path}/{dataset}/first_stage.tsv")]
    # short_dataset_list = [dataset for dataset in short_dataset_list if dataset not in dataset_list]
    # dataset_list = short_dataset_list + dataset_list
    #dataset_list = [dataset for dataset in dataset_list if dataset not in hypo_colbert.keys()]
    
    print(f"Start {dataset_list} reranking.")
    print(f"model path: {model_path}")

    if model_step:
        checkpoint = f"/data/colbert-prf/experiments/{model_path}/train.py/msmarco.psg.cosine/checkpoints/colbert-{model_step}.dnn"
    else:
        checkpoint = f"/data/ColBERT/experiments/checkpoint/{model_path}/pytorch_model.bin"
    if not os.path.isfile(checkpoint):
        checkpoint = ""
        print("@@@ Only load bert-base-uncased model @@@")
    model_ours, checkpoint = load_model(checkpoint=checkpoint, do_print=True)
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

        # Initialize
        qid_to_embs, qid_to_ids, qid_to_masks = {}, {}, {}
        pid_to_embs, pid_to_ids = {}, {}
        
        colbert_scores_list = []
        hypo_colbert[dataset] = {}
        #colbert_sensim[dataset] = {}

        # qids encoding
        start = time.time()
        qids = list(set([qid for qid, _ in qrels[dataset].items()]))
        print(f"\n{dataset}: {len(qids)} qids encoding...")
        texts = [queries[dataset][qid] for qid in qids]
        Q, Q_ids, masks = inference_ours.queryFromText(texts, bsize=512, with_ids=True)
        Q, Q_ids, masks = Q.detach().cpu(), Q_ids.cpu(), masks.cpu()
        for qid, q, q_ids, q_mask in zip(qids, Q, Q_ids, masks):
            qid_to_embs[qid] = q
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
            D, D_ids = inference_ours.docFromText(docs_batch, bsize=512, keep_dims=False, with_ids=True)
            pids_batch = pids_batches[i]
            for pid, d, d_ids in zip(pids_batch, D, D_ids):
                pid_to_embs[pid] = d
                pid_to_ids[pid] = d_ids
        print(f"{dataset}: {len(pids)} pids Done!")
        print(f"pids encoding time: {time.time()-start}s.\n")


        if post_processing == "whitening":
            print("Whitening Start!")
            all_vecs = list(qid_to_embs.values()) + list(pid_to_embs.values())
            kernel, bias = compute_kernel_bias(all_vecs)
            for qid, embs in tqdm(qid_to_embs.items()):
                qid_to_embs[qid] = transform_and_normalize(embs, kernel, bias)
            for pid, embs in tqdm(pid_to_embs.items()):
                pid_to_embs[pid] = transform_and_normalize(embs, kernel, bias)
            print("Whitening Done!")

        elif post_processing == "glow":
            from normalizing_flows.model import Glow
            print("Glow Start!")
            glow_model_path = f"/data/colbert-prf/index/{index_path}/{dataset}/glow_model_bm25.pt"
            if os.path.isfile(glow_model_path):
                print(f"Load Glow checkpoint.")
                glow_model = Glow(model_path=glow_model_path)
            else:
                print(f"Train Glow.")
                glow_model = Glow()
                all_vecs = list(qid_to_embs.values()) + list(pid_to_embs.values())
                all_vecs = torch.cat(all_vecs, dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
                glow_model.train(all_vecs, bsize=1024, save_path=glow_model_path)
                print(f"Train Glow Done.")

            for qid, embs in tqdm(qid_to_embs.items()):
                embs = embs.unsqueeze(dim=-1).unsqueeze(dim=-1)
                qid_to_embs[qid] = F.normalize(glow_model.transform(embs).squeeze(), dim=-1)

            all_vecs = list(pid_to_embs.values())
            all_vecs = torch.cat(all_vecs, dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).to(torch.float32)
            batches = split_into_batches(all_vecs, bsize=2048)
            all_embs = []
            for batch in tqdm(batches):
                embs = F.normalize(glow_model.transform(batch).squeeze(), dim=-1).cpu().to(torch.float16)
                all_embs.append(embs)
            all_embs = torch.cat(all_embs, dim=0)
            start_idx = 0
            for pid, embs in tqdm(pid_to_embs.items()):
                pid_to_embs[pid] = all_embs[start_idx:start_idx+len(embs)]
                start_idx = start_idx+len(embs)
            print("Glow Done!")

        elif post_processing == "nice":
            from pytorch_nice.nice import NICE
            print("NICE Start!")
            nice_model_path = f"/data/colbert-prf/index/{index_path}/{dataset}/nice_model_bm25.pt"
            if os.path.isfile(nice_model_path):
                print(f"Load NICE checkpoint.")
                nice_model = NICE(model_path=nice_model_path)
                nice_model.cuda()
            else:
                print(f"Train NICE.")
                assert False
                nice_model = NICE()
                nice_model.cuda()
                nice_model.train()
                all_vecs = list(qid_to_embs.values()) + list(pid_to_embs.values())
                all_vecs = torch.cat(all_vecs, dim=0)
                nice_model.do_train(all_vecs, bsize=1024, save_path=nice_model_path)
                print(f"Train NICE Done.")

            nice_model.eval()
            with torch.no_grad():
                for qid, embs in tqdm(qid_to_embs.items()):
                    qid_to_embs[qid] = F.normalize(nice_model.transform(embs).squeeze(), dim=-1).cpu().to(torch.float16)

                if dataset in ["msmarco", "hotpotqa"]:
                    for pid, embs in tqdm(pid_to_embs.items()):
                        pid_to_embs[pid] = F.normalize(nice_model.transform(embs).squeeze(), dim=-1).cpu().to(torch.float16)
                else:
                    print("Batch NICE")
                    all_vecs = list(pid_to_embs.values())
                    #all_vecs = torch.cat(all_vecs, dim=0).to(torch.float32)
                    all_vecs = torch.cat(all_vecs, dim=0)
                    batches = split_into_batches(all_vecs, bsize=2048)
                    all_embs = []
                    for batch in tqdm(batches):
                        embs = F.normalize(nice_model.transform(batch), dim=-1).cpu().to(torch.float16)
                        all_embs.append(embs)
                    all_embs = torch.cat(all_embs, dim=0)
                    start_idx = 0
                    for pid, embs in tqdm(pid_to_embs.items()):
                        pid_to_embs[pid] = all_embs[start_idx:start_idx+len(embs)]
                        start_idx = start_idx+len(embs)
            print("NICE Done!")


        score_dist = []
        # compute late-interaction
        for qid, pids in tqdm(qid_to_pids[dataset].items()):
            Q, Q_ids, masks = qid_to_embs[qid], qid_to_ids[qid], qid_to_masks[qid]
            Q, Q_ids, masks = Q.cuda(), Q_ids.cuda(), masks.cuda()
            q_len = torch.sum(masks).item()


            # compute hypothesis
            colbert_tmp = []
            pids_batches = split_into_batches(pids, bsize=512)

            for batch in pids_batches:
                D = [pid_to_embs[pid] for pid in batch]
                D = pad_sequence(D, batch_first=True).cuda()

                # compute hypothesis
                dot = late_interaction(Q, D)
                maxscoreQ, argmaxscoreQ = dot.max(2)
                colbert_tmp.extend(maxscoreQ.cpu().tolist())

                # D_ids = [pid_to_ids[pid] for pid in batch]
                # D_ids = pad_sequence(D_ids, batch_first=True, padding_value=-100).cuda()
                # valid_pid_toks = torch.sum(D_ids != -100, dim=-1)
                # scores = dot[:, :q_len]

                #for score, d_len in zip(scores, valid_pid_toks):
                #    score_dist.append(score[:,:d_len].reshape(-1).to(torch.float16).cpu())

            colbert_scores_list.append(colbert_tmp)
        

        #score_dist = torch.cat(score_dist)
        #score_dist = score_dist[random.sample(range(len(score_dist)), k=10_000)]
        #colbert_sensim[dataset] = score_dist

        assert len(qid_to_pids[dataset]) == len(colbert_scores_list)
        for (qid, pids), colbert_scores in zip(tqdm(qid_to_pids[dataset].items()), colbert_scores_list):
            hypo_colbert[dataset][qid] = {}

            assert len(colbert_scores) == len(pids), f"{len(colbert_scores)}, {len(pids)}"
            for pid, colbert_score in zip(pids, colbert_scores):
                colbert_score = sum(colbert_score)
                hypo_colbert[dataset][qid][pid] = colbert_score

        ndcg, mrr = evaluation(hypo_colbert[dataset], qrels[dataset])
        print(f"{model_path} {dataset} {post_processing} ndcg@10: {ndcg} mrr@10: {mrr}")

        if (idx%10 == 9) or (idx == len(dataset_list)-1):
            pass
            # pickle_dump(hypo_colbert, f"{save_path}/hypo_colbert{bm25_filename}.pickle")
            # pickle_dump(colbert_sensim, f"{save_path}/colbert_score_dist.pickle")
            # print(f"Saving {model_path} {dataset}, {len(hypo_colbert)} are saved!")


        #sensim_score = torch.mean(colbert_sensim[dataset]).item()
        #print(f"{model_path} {dataset} sensim_score: {round(sensim_score,6)}")
        print(f"{model_path} {dataset} done.")