#!/usr/bin/env python3

import argparse, json, os, time
from collections import defaultdict

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from scipy.special import digamma
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
#                     Helper functions
# ────────────────────────────────────────────────────────────────
def get_doc_text(docid, searcher, *, truncate_len=512):
    """
    Extract "title" + "text" field from a BEIR multifield document
    and truncate to `truncate_len` word pieces.
    """
    try:
        raw = searcher.doc(docid).raw()
        title = raw.split('"title" : "')[-1].split('"text')[0].rstrip('", \n')
        body = raw.split('"text" : "')[-1].split('"metadata')[0].rstrip('", \n')
        full = f"{title} {body}".strip()
        return " ".join(full.split()[:truncate_len])
    except AttributeError:          # deleted / corrupted doc
        return None


def collect_topics(searcher, truncate_len=512):
    """
    Treat every document as a query; return {docid: truncated_text}.
    """
    topics = {}
    for i in range(searcher.num_docs):
        did = searcher.doc(i).docid()
        txt = get_doc_text(did, searcher, truncate_len=truncate_len)
        if txt:
            topics[did] = txt
    return topics


def write_topics_tsv(path, topics):
    with open(path, "w") as f:
        for did, text in topics.items():
            f.write(f"{did}\t{text}\n")


def run_bm25_search(index_name, topics_tsv, out_path, k_max, batch=128, threads=16):
    """
    Wrapper around Pyserini CLI for maximum compatibility with existing code.
    """
    cmd = (
        "python -m pyserini.search.lucene"
        f" --index {index_name}"
        f" --topics {topics_tsv}"
        f" --output {out_path}"
        " --output-format trec"
        f" --batch {batch} --threads {threads}"
        f" --hits {k_max}"
    )
    t0 = time.time()
    os.system(cmd)
    print(f"[INFO] Retrieval finished in {time.time() - t0:,.1f} s")


def read_trec_run(path):
    """
    Load TREC run file → {qid: [(docid, score), …]}.
    """
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, docid, _, score, _ = line.split()
            data[qid].append((docid, float(score)))
    return data


# ────────────────────────────────────────────────────────────────
#              Knee detection on the k-distance curve
# ────────────────────────────────────────────────────────────────
def knee_point(k_vals, med_scores):
    """
    Simple "maximum distance to chord" elbow locator.
    Returns the index (1-based) of the knee; falls back to 5 if detection fails.
    """
    # line through first and last points: p0 (k1, y1) → p1 (kL, yL)
    k1, y1 = k_vals[0], med_scores[0]
    kL, yL = k_vals[-1], med_scores[-1]

    # normalised perpendicular distance of each point to the chord
    denom = np.hypot(kL - k1, yL - y1)
    if denom == 0:
        return 5

    dist = []
    for k, y in zip(k_vals, med_scores):
        # area of triangle ×2 / base length  == |(y1 − yL)·k + (kL − k1)·y + (k1·yL − kL·y1)| / base
        d = abs((y1 - yL) * k + (kL - k1) * y + (k1 * yL - kL * y1)) / denom
        dist.append(d)

    knee_idx = int(np.argmax(dist))          # 0-based
    return k_vals[knee_idx]


def choose_optimal_k(run_data, k_max):
    """
    Build μ_k curve (median score of the k-th neighbour) and pick the knee.
    """
    kth_scores = [[] for _ in range(k_max)]

    for hits in run_data.values():
        hits = sorted(hits, key=lambda x: x[1], reverse=True)
        for k in range(1, k_max + 1):
            score = hits[k - 1][1] if len(hits) >= k else 0.0
            kth_scores[k - 1].append(score)

    med_scores = [float(np.median(s)) for s in kth_scores]
    k_vals = list(range(1, k_max + 1))
    k_star = knee_point(k_vals, med_scores)

    print("[INFO] k-distance medians:", ", ".join(f"{m:.4g}" for m in med_scores[:10]), "...")
    print(f"[INFO] Knee detected at k = {k_star}")
    return k_star


# ────────────────────────────────────────────────────────────────
#                 KL-entropy (Cressie–Read) estimator
# ────────────────────────────────────────────────────────────────
def kl_entropy_one(hits, *, k, N, delta=1e-6):
    """
    EPS-k estimator (Wang et al., 2021 variant) using 1 / score_k as ε_k.
    Returns None if fewer than k neighbours are present.
    """
    if len(hits) < k:
        return None
    nn_score = sorted(hits, key=lambda x: x[1], reverse=True)[k - 1][1]
    eps_k = 1.0 / (nn_score + delta)          # distance proxy
    return digamma(N) - digamma(k) + np.log(eps_k)


# ────────────────────────────────────────────────────────────────
#                         Main pipeline
# ────────────────────────────────────────────────────────────────
def main(args):
    index_name = f"beir-v1.0.0-{args.dataset_name}.multifield"
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    N = searcher.num_docs
    os.makedirs(args.intermediate_dir, exist_ok=True)
    os.makedirs(args.save_entropy_dict_dir, exist_ok=True)

    # 1. prepare topics
    topics = collect_topics(searcher, truncate_len=512)
    topics_tsv = os.path.join(args.intermediate_dir, f"topics_{args.dataset_name}.tsv")
    write_topics_tsv(topics_tsv, topics)

    # 2. retrieve top-k_max neighbours
    run_path = os.path.join(args.intermediate_dir, f"bm25_{args.dataset_name}.trec")
    run_bm25_search(index_name, topics_tsv, run_path, args.k_max)

    run_data = read_trec_run(run_path)

    # 3. choose k automatically (or use user-given k)
    if args.k is None:
        k_star = choose_optimal_k(run_data, args.k_max)
    else:
        k_star = args.k
        print(f"[INFO] Using user-supplied k = {k_star}")

    # 4. compute KL-entropy table
    entropy_tbl = {}
    for qid, hits in tqdm(run_data.items(), desc="KL-entropy"):
        h = kl_entropy_one(hits, k=k_star, N=N)
        if h is not None:
            entropy_tbl[qid] = h

    out_file = os.path.join(args.save_entropy_dict_dir, f"klentropy_k{k_star}.json")
    with open(out_file, "w") as f:
        json.dump(entropy_tbl, f)

    print(f"[INFO] Saved {len(entropy_tbl):,} entropy values → {out_file}")


# ────────────────────────────────────────────────────────────────
#                         CLI arguments
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, help="BEIR subset (e.g. scifact)")
    p.add_argument("--save_entropy_dict_dir", required=True, help="output dir")
    p.add_argument("--intermediate_dir", default="./tmp", help="scratch dir")
    p.add_argument("--k_max", type=int, default=10,
                   help="retrieve this many neighbours once (should be ≳ expected knee)")
    p.add_argument("--k", type=int, default=None,
                   help="override automatic knee detection with a fixed k")
    main(p.parse_args())
