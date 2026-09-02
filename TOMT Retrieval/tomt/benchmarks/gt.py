import os

from tomt.data.utils import read_jsonl, read_qrels, read_json


def get_documents(folder_path, hard_negatives=False, negatives=False, return_type_dict=False):
    fpath = os.path.join(folder_path, "documents.json")
    documents = read_jsonl(fpath)
    type_dict = {d["id"]: "pos" for d in documents}

    if negatives:
        neg_docs = read_jsonl(os.path.join(folder_path, "negative_documents.json"))
        documents.extend(neg_docs)
        type_dict.update({d["id"]: "neg" for d in neg_docs})

    if hard_negatives:
        hard_neg = read_jsonl(os.path.join(folder_path, "hard_negative_documents.json"))
        documents.extend(hard_neg)
        type_dict.update({d["id"]: "hard_neg" for d in hard_neg})

    if return_type_dict:
        return documents, type_dict
    return documents


class GTData:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self._ids_by_subset = {
            "all": set(self._queries_to_dict(self.get_queries()).keys())
        }

    @property
    def available_subsets(self):
        return set(self._ids_by_subset.keys())

    def _queries_to_dict(self, queries):
        d = {}
        for query in queries:
            d[query["id"]] = query
        return d

    def get_queries(self):
        fpath = os.path.join(self.folder_path, "queries.json")
        return read_jsonl(fpath)

    def get_qrels(self, for_pytrec=True):
        fpath = os.path.join(self.folder_path, "qrels.txt")
        return read_qrels(fpath, for_pytrec)

    def get_ids_by_subset(self, subset):
        return self._ids_by_subset[subset]

    def get_grouped_data(self, dataset_root, negative_set, hn_source="bm25"):
        assert hn_source in {"bm25", "tomt_hn"}
        hard_negatives = negative_set in {"hn", "all"}
        negatives = negative_set in {"neg", "all"}
        documents = get_documents(dataset_root, hard_negatives=hard_negatives, negatives=negatives)
        documents = {d["id"]: d for d in documents}

        bm25_negatives = read_json(os.path.join(dataset_root, f"bm25_hard_negatives_{negative_set}.json"))
        hn_source_map = {}
        if hn_source == "bm25":
            negatives_key_name = "bm25_negatives"
            query_hard_negatives = bm25_negatives

            for qid in bm25_negatives:
                hn_source_map[qid] = "bm25"
        else:
            negatives_key_name = "bm25_hn_negatives"
            assert negative_set == "all", "This is viable only if negative_set is 'all'"
            neg_ids = read_json(os.path.join(dataset_root, "sub_id_to_neg_doc_ids.json"))
            queries = self.get_queries()
            queries = {q["id"]: q for q in queries}

            query_hard_negatives = {}
            for qid in queries:
                if qid not in queries:
                    continue
                neg = neg_ids.get(qid, [])
                neg = [n for n in neg if n in documents]
                if len(neg) > 0:
                    query_hard_negatives[qid] = neg
                    hn_source_map[qid] = "tomt_hn"
                else:
                    query_hard_negatives[qid] = bm25_negatives[qid]
                    hn_source_map[qid] = "bm25"

        queries = self.get_queries()
        qrels = self.get_qrels(False)

        data = []
        for query in queries:

            gold_doc_id = qrels[query["id"]]
            positive_documents = [documents[gold_doc_id]]

            negs = []
            for doc_id in query_hard_negatives[query["id"]]:
                negs.append(documents[doc_id])

            data.append({
                "id": query["id"],
                "query": query,
                "positive_documents": positive_documents,
                negatives_key_name: negs,
                "hn_source": hn_source_map[query["id"]]
            })

        return data
