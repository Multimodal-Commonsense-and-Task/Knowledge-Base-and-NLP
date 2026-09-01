import os
import math
import torch
import pickle
import ujson

from tqdm.auto import tqdm
from transformers import BertTokenizerFast


def pickle_dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def split_into_batches(batch, bsize):
    batches = []
    for offset in range(0, len(batch), bsize):
        batches.append(batch[offset:offset+bsize])
    return batches

def read_trec_corpus(path):
    corpus = []

    with open(path, "r") as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            corpus.append(passage)
    
    return corpus

def read_corpus(path):
    corpus = []

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            data = ujson.loads(line)
            if "_id" not in data:
                _id = data["id"]
            else:
                _id = data["_id"]

            title = data["title"].strip()
            text = data["text"].strip()
            corpus.append(" ".join([title, text]))

    return corpus

def read_queries(path):
    queries = []

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            data = ujson.loads(line)
            if "_id" not in data:
                _id = data["id"]
            else:
                _id = data["_id"]
            text = data["text"].strip()
            queries.append(text)

    return queries

def tokenize(corpus, isdoc=False):
    tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
    Q_marker_token_id = tok.convert_tokens_to_ids('[unused0]')
    D_marker_token_id = tok.convert_tokens_to_ids('[unused1]')
    cls_token_id = tok.cls_token_id
    sep_token_id = tok.sep_token_id
    mask_token_id = tok.mask_token_id
    query_maxlen = 32

    batches = split_into_batches(corpus, bsize=512)
    vocab = {}
    for batch_text in tqdm(batches):
        #tokens = [tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        tokens = [tok(x, add_special_tokens=False)['input_ids'] for x in batch_text]
        if isdoc:
            prefix, suffix = [cls_token_id, D_marker_token_id], [sep_token_id]
            tokens = [prefix + lst + suffix for lst in tokens]
        else:
            prefix, suffix = [cls_token_id, Q_marker_token_id], [sep_token_id]
            tokens = [prefix + lst + suffix + [mask_token_id] * (query_maxlen - (len(lst)+3)) for lst in tokens]

        for token in tokens:
            token = set(token)
            for t in token:
                if t not in vocab.keys():
                    vocab[t] = 0
                vocab[t] += 1

    idf = {t: math.log(len(corpus)/(1+cnt)) for t, cnt in vocab.items()}
    return idf


if __name__ == "__main__":
    dataset_list = ["bioasq"]
    
    for dataset in dataset_list:
        print(f"Start dataset: {dataset}")
        queries = read_queries(f"/data/beir/{dataset}/queries.jsonl")
        save_path = f"/data/beir/{dataset}"

        qidf = tokenize(queries, isdoc=False)
        print(f"{save_path}: qidf Tokenizing Done.")
        pickle_dump(qidf, os.path.join(save_path, "qidf.pickle"))
        print(f"{save_path}: qidf Save Done.")


        corpus = read_corpus(f"/data/beir/{dataset}/corpus.jsonl")
        save_path = f"/data/beir/{dataset}"

        idf = tokenize(corpus, isdoc=True)
        print(f"{save_path}: idf Tokenizing Done.")
        pickle_dump(idf, os.path.join(save_path, "idf.pickle"))
        print(f"{save_path}: idf Save Done.")
