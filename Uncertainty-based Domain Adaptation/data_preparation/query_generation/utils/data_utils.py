import os, pathlib
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import spacy
from spacy.tokens import DocBin
import json

out_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), "data")


def load_dataset(dataset):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    data_path = util.download_and_unzip(url, out_dir)
    return GenericDataLoader(data_path).load(split="test")


def load_spacy_data(dataset):
    docbin_path = os.path.join(out_dir, f"{dataset}/{dataset}_spacy.bin")
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(docbin_path)
    return list(doc_bin.get_docs(nlp.vocab))


def concat_title_and_body(did, corpus, sep):
    document = []
    title = corpus[did]["title"].strip()
    body = corpus[did]["text"].strip()
    if len(title):
        document.append(title)
    if len(body):
        document.append(body)
    return sep.join(document)


def load_jsonl(dataset):
    return [json.loads(line) for line in open(f"{out_dir}/{dataset}-sampled.jsonl")]


def load_jsonl_corpus(dataset, case=""):
    data = {}
    with open(f"{out_dir}/{dataset}_sampled{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["doctext"]
    return data


def load_jsonl_summary(dataset, case=""):
    data = {}
    with open(f"{out_dir}/{dataset}_summary{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["summary"]
    return data


def load_jsonl_clarified(dataset, case=""):
    data = {}
    with open(f"{out_dir}/{dataset}_clarified{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["summary"]
    return data


def load_jsonl_streamlined(dataset, case=""):
    data = {}
    with open(f"{out_dir}/{dataset}_streamlined{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["summary"]
    return data


def load_jsonl_keyword(dataset, case=""):
    data = {}
    with open(f"{out_dir}/{dataset}_keyword{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["summary"]
    return data


def load_jsonl_nearest(dataset, case=""):
    data = {}
    nearest = {}
    with open(f"{out_dir}/{dataset}_nearest{case}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            data[e["docid"]] = e["doctext"]
            nearest[e["docid"]] = e["nearest_docs"]
    return data, nearest


def load_jsonl_cache(dataset, type="query"):
    data = {}
    if os.path.exists(f"{out_dir}/{dataset}_{type}.jsonl.cache"):
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache") as f:
            for line in f:
                e = json.loads(line)
                data[e["docid"]] = e["question"]
    else:
        with open(f"{out_dir}/{dataset}_{type}.jsonl", "w") as f:
            pass
    return data


def load_jsonl_summary_cache(dataset, type="query_summary"):
    summary = {}
    data = {}
    if os.path.exists(f"{out_dir}/{dataset}_{type}.jsonl.cache"):
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache") as f:
            for line in f:
                e = json.loads(line)
                summary[e["docid"]] = e["summary"]
                data[e["docid"]] = e["question"]
    else:
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache", "w") as f:
            pass
    return summary, data


def load_jsonl_summary_only_cache(dataset, type="summary"):
    data = {}
    if os.path.exists(f"{out_dir}/{dataset}_{type}.jsonl.cache"):
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache") as f:
            for line in f:
                e = json.loads(line)
                data[e["docid"]] = e["summary"]
    else:
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache", "w") as f:
            pass
    return data


def load_jsonl_decision_only_cache(dataset, type="decision"):
    data = {}
    if os.path.exists(f"{out_dir}/{dataset}_{type}.jsonl.cache"):
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache") as f:
            for line in f:
                e = json.loads(line)
                data[e["docid"]] = e["decision"]
    else:
        with open(f"{out_dir}/{dataset}_{type}.jsonl.cache", "w") as f:
            pass
    return data
