import os
from argparse import Namespace
from haystack.document_store import InMemoryDocumentStore
from haystack.preprocessor import PreProcessor
from tomt.benchmarks.gt import GTData, get_documents
from tomt.benchmarks.lexical import initialize_from_config as init_lexical
from run_lexical_benchmark import make_query, get_data_path, read_config, prepare_documents, process_batch
from tqdm import tqdm
from tomt.data import utils


def process(batch):
    query_batch = [(q["meta"]["id"], q["text"]) for q in batch]
    res = retriever.batch_retrieve(query_batch)
    for qid, scores in res.items():
        i = 0
        gold_doc = qrels[q_dict["id"]]
        while len(hn) != method["n_hard_negatives"]:
            doc, score = scores[i]
            i += 1
            if doc.id == gold_doc:
                continue
            hn.append(doc.id)

        hard_negatives[qid] = hn


if __name__ == '__main__':

    method = {
        "method": "terrier",
        "config_path": "./config/lexical/terrier_bm25.json",
        "common_index_path": "./common_index",
        "n_hard_negatives": 5,
        "negative_set": "all"
    }

    data_root = "./dataset/"
    neg_set = method["negative_set"]

    for dataset in ["Movies", "Books"]:

        for split in ["test"]:
            queries = GTData(os.path.join(data_root, dataset, "splits", split)).get_grouped_data(
                os.path.join(data_root, dataset), "all")
            print(queries[0])

        break

        folder_path = get_data_path(dataset)

        processor = PreProcessor(clean_empty_lines=True,
                                 clean_whitespace=True,
                                 clean_header_footer=True,
                                 split_by=None,
                                 split_respect_sentence_boundary=True)

        config_json = read_config(method["config_path"])
        document_store = InMemoryDocumentStore()
        documents, processed_docs = prepare_documents(Namespace(negative_set=neg_set), folder_path, processor)
        document_store.write_documents(processed_docs)
        config_json[
            "index_path"] = os.path.join(method["common_index_path"], "terrier", f"{dataset}_{neg_set}")
        retriever = init_lexical("terrier", method["n_hard_negatives"] + 1, document_store, config_json)

        gtdata = GTData(os.path.join(data_root, dataset))
        queries = gtdata.get_queries()
        qrels = gtdata.get_qrels(False)

        batch = []
        batch_size = 100
        hard_negatives = {}
        for q_dict in tqdm(queries):

            q = processor.process(make_query(q_dict, "all"))[0]
            hn = []
            batch.append(q)
            if len(batch) >= batch_size:
                process(batch)
                batch = []

        if len(batch) > 0:
            process(batch)
            batch = []

        utils.write_json(hard_negatives, os.path.join(data_root, dataset, f"bm25_hard_negatives_{neg_set}.json"))
