import os
import ujson
import torch
import pickle
import numpy as np
import pandas as pd


def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pickle_dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def flatten_2d(L):
    return [x for y in L for x in y]

def flatten_3d(L):
    return [x for z in L for y in z for x in y]

def split_into_batches(batch, bsize):
    batches = []
    for offset in range(0, len(batch), bsize):
        batches.append(batch[offset:offset+bsize])
    return batches

def split_into_batches_sort(batch, bsize):
    import transformers
    assert isinstance(batch, transformers.tokenization_utils_base.BatchEncoding) or isinstance(batch, dict)
    
    max_length = torch.sum(batch["attention_mask"], dim=-1).detach().cpu()  # (512,)
    values, indices = torch.sort(max_length)
    batch = {k: v[indices].detach().cpu() for k, v in batch.items()}
    reverse_indices = indices.sort().indices

    batches = [dict() for i, offset in enumerate(range(0, len(batch["input_ids"]), bsize))]
    max_lengthes = []
    for idx, offset in enumerate(range(0, len(batch["attention_mask"]), bsize)):
        att = batch["attention_mask"][offset : offset + bsize]
        max_length = torch.max(torch.sum(att, dim=-1)).item()
        max_lengthes.append(max_length)

    for k, v in batch.items():
        for idx, (offset, max_length) in enumerate(zip(range(0, len(v), bsize), max_lengthes)):
            batches[idx][k] = v[offset : offset + bsize][:, :max_length]
    return batches, reverse_indices


def read_first_stage(path, cbpid_to_id):
    qid_to_lx_pids = {}
    qid_to_cb_pids = {}
    qid_to_ours_pids = {}

    df = pd.read_csv(path, sep="\t", names=["qid", "pid", "islx", "iscb"])
    for qid, pid, islx, iscb in zip(df["qid"], df["pid"], df["islx"], df["iscb"]):
        qid = str(qid)
        pid = cbpid_to_id[pid]
        
        if qid not in qid_to_lx_pids and islx is True:
            qid_to_lx_pids[qid] = []
        if qid not in qid_to_cb_pids and iscb is True:
            qid_to_cb_pids[qid] = []
        if qid not in qid_to_ours_pids:
            qid_to_ours_pids[qid] = []

        if islx:
            qid_to_lx_pids[qid].append(pid)
        if iscb:
            qid_to_cb_pids[qid].append(pid)
        qid_to_ours_pids[qid].append(pid)

    qid_to_lx_pids = {qid:list(set(pids)) for qid, pids in qid_to_lx_pids.items()}
    qid_to_cb_pids = {qid:list(set(pids)) for qid, pids in qid_to_cb_pids.items()}
    qid_to_ours_pids = {qid:list(set(pids)) for qid, pids in qid_to_ours_pids.items()}

    return qid_to_lx_pids, qid_to_cb_pids, qid_to_ours_pids


def read_corpus(path, valid_pids=None):
    pid_to_doc = {}

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            data = ujson.loads(line)
            if "id" in data:
                data["_id"] = data["id"]
            _id = str(data["_id"])
            if (valid_pids is not None) and (_id not in valid_pids):
                continue
            title = data["title"].strip()
            text = data["text"].strip()
            if len(title) == 0 and len(text) == 0:
                continue
            pid_to_doc[_id] = " ".join([title, text])
    return pid_to_doc

def read_corpus_dpr_scale(path, valid_pids=None):
    corpus = {}
    with open(path, "r") as f:
        for line in f:
            row = ujson.loads(line)
            if "id" in row:
                row["_id"] = row["id"]
            _id = str(row["_id"])
            if (valid_pids is not None) and (_id not in valid_pids):
                continue
            title = row["title"]
            text = row["text"]
            text = maybe_add_title(text, title, use_title=True, sep_token=" ")
            
            assert _id not in corpus, _id
            corpus[_id] = text
    return corpus

def read_bm25_results(path):
    qid_to_bm25 = {}
    with open(path, "r") as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split(" ")
            if qid not in qid_to_bm25:
                qid_to_bm25[qid] = []
            qid_to_bm25[qid].append(pid)
    return qid_to_bm25

def read_bm25_results_with_score(path):
    qid_to_bm25 = {}
    with open(path, "r") as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split(" ")
            if qid not in qid_to_bm25:
                qid_to_bm25[qid] = {}
            qid_to_bm25[qid][pid] = float(score)
    return qid_to_bm25

def read_queries_tsv(path, qrels=None):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            qid, text = line.split("\t")
            text = text.strip()
            if qrels is not None and qid not in qrels:
                continue
            assert qid not in queries, qid
            queries[qid] = text
    return queries

def read_queries(path, qrels=None):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            data = ujson.loads(line)
            if "_id" not in data:
                _id = data["id"]
            else:
                _id = data["_id"]
            text = data["text"]
            if qrels is not None and _id not in qrels:
                continue
            assert _id not in queries, _id
            queries[_id] = text
    return queries

def read_qrels(path):
    qrels = {}
    with open(path, "r") as f:
        for line in f:
            qid, pid, score = line.strip().split("\t")
            if score == "score":
                continue
            if qid not in qrels:
                qrels[qid] = {}
            assert pid not in qrels[qid], pid
            qrels[qid][pid] = int(score)
    return qrels

def maybe_add_title(text, title, use_title, sep_token):
    if use_title and len(title) > 0:
        return " ".join([title, sep_token, text])
    else:
        return text

def load_model(checkpoint, qidf=None, do_print=True):
    from colbert.modeling.inference import ModelInference
    from colbert.modeling.colbert import ColBERT
    from colbert.parameters import DEVICE
    from colbert.utils.utils import print_message, load_checkpoint
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                        query_maxlen=32,
                                        doc_maxlen=180,
                                        dim=128,
                                        similarity_metric="cosine",
                                        mask_punctuation=False,
                                        qidf=qidf,
                                    )
    colbert = colbert.to(DEVICE)
    print_message("#> Loading model checkpoint.", condition=do_print)

    if len(checkpoint) > 0:
        checkpoint = load_checkpoint(checkpoint, colbert, do_print=do_print)
    else:
        print(f"@@@ Loaded only bert.")

    colbert.eval()
    return colbert, checkpoint

def evaluation(hypothesis, qrels, n=10):
    import pytrec_eval
    
    # Evaluation
    ndcg = {}
    mrr = {}

    k_values = [n]
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        mrr[f"MRR@{k}"] = 0.0
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    mrr_string = "recip_rank_" + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, mrr_string})
    scores = evaluator.evaluate(hypothesis)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            mrr[f"MRR@{k}"] += scores[query_id]["recip_rank"]
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"]/len(scores), 5)

    return ndcg[f"NDCG@{n}"], mrr[f"MRR@{n}"]



# Whitening
def compute_kernel_bias(vecs):
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    W, mu = torch.tensor(W).to(torch.float32), torch.tensor(mu).to(torch.float32)
    return W, -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        embs = vecs + bias
        vecs = embs @ kernel
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


if __name__ == "__main__":
    read_first_stage(f"/data/colbert-prf/index-lx-idf-200k-max/arguana/first_stage.tsv")