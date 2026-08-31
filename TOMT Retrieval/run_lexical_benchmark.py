import argparse
import json
import logging.handlers
import os
from collections import defaultdict

import time
import numpy as np
import pytrec_eval
from haystack.document_store import ElasticsearchDocumentStore
from haystack.document_store import InMemoryDocumentStore
from haystack.preprocessor import PreProcessor
from collections import namedtuple
from config import configure_logging
from tomt.benchmarks.gt import GTData, get_documents
from tomt.benchmarks.lexical import initialize_from_config as init_lexical
from tomt.data import utils

USE_CACHE = True
Query = namedtuple("Query", ["text", "id"])

log = logging.getLogger(__name__)


def make_query(query, query_type):
    if query_type == "title_only":
        text = query["title"]
    elif query_type == "description_only":
        text = query["description"]
    else:
        text = query["title"] + "\n" + query["description"]

    return {
        "text": text,
        "meta": query
    }


def add_scores(qid, scores, resulting_qrels):
    resulting_qrels[qid] = {}
    for doc, score in scores:
        resulting_qrels[qid][doc.id] = score


def process_batch(batch, resulting_qrels):
    query_batch = [(q["meta"]["id"], q["text"]) for q in batch]
    res = retriever.batch_retrieve(query_batch)
    for qid, scores in res.items():
        add_scores(qid, scores, resulting_qrels)


def get_data_path(dataset):
    if dataset == "Movies":
        folder_path = os.path.join("./dataset/Movies")
    elif dataset == "Books":
        folder_path = os.path.join("./dataset/Books")
    elif dataset == "TestMovies":
        folder_path = os.path.join("./dataset/TestMovies")
    else:
        raise ValueError()
    return folder_path


def read_config(config_path):
    log.info(f"Reading config from {config_path}")
    config_json = utils.read_json(config_path)

    for k, v in config_json.items():
        log.info(f"\t{k:<10} :: {v}")
    return config_json


def prepare_documents(args, folder_path, processor):
    # get documents
    hard_negatives = args.negative_set in {"hn", "all"}
    negatives = args.negative_set in {"neg", "all"}
    documents = get_documents(os.path.join(folder_path), negatives=negatives, hard_negatives=hard_negatives)

    processed_docs = []
    for doc in documents:
        processed_docs.extend(processor.process(doc))

    return documents, processed_docs


def load_train_val_data(folder_path):
    # merge train/val set and pick hyperparams based on this new set
    val_data = GTData(os.path.join(folder_path, "splits", "validation"))
    val_queries = val_data.get_queries()
    val_qrel = val_data.get_qrels()

    train_data = GTData(os.path.join(folder_path, "splits", "train"))
    train_queries = train_data.get_queries()
    train_qrel = train_data.get_qrels()

    queries = []
    queries.extend(train_queries)
    queries.extend(val_queries)

    qrel = {}
    qrel.update(train_qrel)
    qrel.update(val_qrel)

    available_subsets = val_data.available_subsets
    available_subsets = train_data.available_subsets.union(available_subsets)

    ids_by_subsets = {}
    for subset in available_subsets:
        ids_by_subsets[subset] = val_data.get_ids_by_subset(subset)
        ids_by_subsets[subset] = ids_by_subsets[subset].union(train_data.get_ids_by_subset(subset))

    assert len(val_qrel) + len(train_qrel) == len(qrel)
    assert len(val_queries) + len(train_queries) == len(queries)
    assert len(queries) == len(qrel)

    return queries, qrel, ids_by_subsets


def load_test_data(folder_path):
    data = GTData(os.path.join(folder_path, "splits", "test"))
    queries = data.get_queries()
    qrel = data.get_qrels()
    available_subsets = data.available_subsets
    ids_by_subsets = {}
    for subset in available_subsets:
        ids_by_subsets[subset] = data.get_ids_by_subset(subset)

    return queries, qrel, ids_by_subsets


def add_common_args(parser):
    parser.add_argument("--top_k", type=int, default=1000, help="number of documents to retreive")
    parser.add_argument("--query_type", choices={"title_only", "description_only", "all"}, required=True,
                        help="what info to include when querying")
    neg_set = "none:only pos docs indexed, hn:hard negatives only, neg:negatives only (no hn), all: all of the above"
    parser.add_argument("--negative_set", choices={"none", "hn", "neg", "all"}, required=True,
                        help=f"set of 'negative' documents to index ({neg_set})")

    parser.add_argument("--dataset", choices={"Movies", "Books", "TestMovies"}, required=True,
                        help="Dataset to test on")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", required=True, help="path to config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_lexical_benchmark")
    parser.add_argument("phase", choices={"fit", "evaluate_test"})

    parser.add_argument("--method",
                        required=True,
                        help="Method to benchmark")

    parser.add_argument("--es_host", help="Host location for ElasticSearch (only required for elastic_bm25)")
    parser.add_argument("--es_port", type=int, help="Host location for ElasticSearch (only required for elastic_bm25)")

    parser.add_argument("--out", required=True, help="Location to save results")
    parser.add_argument("--common_index_path", required=True,
                        help="Location of common_index")
    add_common_args(parser)
    args = parser.parse_args()

    assert not os.path.exists(args.out), f"folder {args.out} already exists"

    configure_logging(f"benchmark-{args.method}", args.verbose)

    log = logging.getLogger("run_benchmark")

    folder_path = get_data_path(args.dataset)

    start_time = time.time()
    processor = PreProcessor(clean_empty_lines=True,
                             clean_whitespace=True,
                             clean_header_footer=True,
                             split_by=None,
                             split_respect_sentence_boundary=True)

    config_json = read_config(args.config)

    if args.method.startswith("elastic"):
        assert args.es_host is not None and args.es_port is not None, "ES args needed"
        document_store = ElasticsearchDocumentStore(args.es_host, args.es_port)
        # clear out the index
        document_store.delete_all_documents()
    else:
        document_store = InMemoryDocumentStore()

    documents, processed_docs = prepare_documents(args, folder_path, processor)
    document_store.write_documents(processed_docs)

    # terrier requires a path to an index
    if args.method.startswith("terrier"):
        config_json[
            "index_path"] = os.path.join(args.common_index_path, "terrier", f"{args.dataset}_{args.negative_set}")
    retriever = init_lexical(args.method, args.top_k, document_store, config_json)

    if args.phase == "fit":
        queries, qrel, ids_by_subsets = load_train_val_data(folder_path)
    else:
        queries, qrel, ids_by_subsets = load_test_data(folder_path)

    log.info(f"Phase {args.phase}: {len(queries)}")
    subset_metrics = {}

    # process queries
    processed_queries = []
    resulting_qrels = {}

    batch = []
    batch_size = 1000
    successes = []
    for i, q_dict in enumerate(queries, 1):
        if i % 100 == 0:
            log.info(f"\t{i}/{len(queries)} done.")

        q = processor.process(make_query(q_dict, args.query_type))[0]
        successes.append(q_dict["id"])
        batch.append(q)
        if len(batch) >= batch_size:
            process_batch(batch, resulting_qrels)
            batch = []

    if len(batch) > 0:
        process_batch(batch, resulting_qrels)
        batch = []

    log.info(f"Run took {time.time() - start_time:0.4f}s")
    log.info(f"N successes:: {len(successes)} out of {len(queries)}")
    metrics_to_compute = {'recip_rank', 'recall_10', "recall_1"}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, metrics_to_compute)

    acc_metrics = defaultdict(list)
    for qid, res in evaluator.evaluate(resulting_qrels).items():
        for met in metrics_to_compute:
            acc_metrics[met].append((qid, res[met]))

    mean_metrics = {}
    for subset, subset_ids in ids_by_subsets.items():
        log.info(f"Metrics: {subset}")
        mean_metrics[subset] = {}
        for met, all_vals in sorted(acc_metrics.items(), key=lambda _: _[0]):
            vals = [val for (qid, val) in all_vals if qid in subset_ids]
            mean_metrics[subset][met] = {
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
            log.info(f"\t{subset}:: {met}: {round(np.mean(vals), 4)}, ({round(np.std(vals), 4)})")

    os.makedirs(args.out, exist_ok=False)

    utils.write_json(vars(args), os.path.join(args.out, "args.json"), indent=2)
    utils.write_json(mean_metrics, os.path.join(args.out, "metrics.json"), indent=2)
    utils.write_json(resulting_qrels, os.path.join(args.out, "out_qrels.json.gzip"), zipped=True)

    log.info(f"Run took {time.time() - start_time:0.4f}s")
