import json
import logging
import os

import numpy as np

from tomt.data.utils import read_jsonl, read_qrels

log = logging.getLogger(__name__)


def train_val_test_split(folder_path, val_prop=0.1, test_prop=0.1, seed=42):
    queries = read_jsonl(os.path.join(folder_path, "raw_queries.json"))
    log.info(f"Got {len(queries)} queries")

    np.random.seed(seed)
    val_size = int(val_prop * len(queries))
    test_size = int(test_prop * len(queries))
    train_size = len(queries) - test_size - val_size

    shuffle_idx = np.random.permutation(np.arange(len(queries)))
    train_idx = shuffle_idx[:train_size]
    val_idx = shuffle_idx[train_size: train_size + val_size]
    test_idx = shuffle_idx[train_size + val_size:]

    idx = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx
    }

    log.info(f"Train: {len(train_idx)} ({train_size})")
    log.info(f"Validation: {len(val_idx)} ({val_size})")
    log.info(f"Test: {len(test_idx)} ({test_size})")

    assert train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0] == len(queries)

    os.makedirs(os.path.join(folder_path, "splits"), exist_ok=False)
    qrels = read_qrels(os.path.join(folder_path, "qrels.txt"), for_pytrec=False)

    for split, indices in idx.items():
        os.makedirs(os.path.join(folder_path, "splits", split), exist_ok=False)
        q_path = os.path.join(folder_path, "splits", split, "raw_queries.json")
        qrel_path = os.path.join(folder_path, "splits", split, "qrels.txt")

        log.info(f"Writing {len(indices)} queries to: {q_path} and {qrel_path}")
        with open(q_path, "w") as query_writer, open(qrel_path, "w") as qrel_writer:
            for i in indices:
                query = queries[i]
                qid = query["id"]
                qrel_writer.write(f"{qid}\t0\t{qrels[qid]}\t{1}\n")
                query_writer.write(json.dumps(query) + "\n")
