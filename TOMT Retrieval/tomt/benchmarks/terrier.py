import logging
from collections import defaultdict
from typing import List, Optional, Tuple
import shutil
import pandas as pd
import pyterrier as pt
from haystack.retriever.base import BaseRetriever
import os
from tomt.benchmarks.lexical_utils import get_std_utils, Utils

if not pt.started():
    pt.init()

log = logging.getLogger(__name__)


class TerrierIndex:
    def __init__(self, index):
        self.i = pt.IndexFactory.of(index)
        self.lex = self.i.getLexicon()

        self.meta = self.i.getMetaIndex()
        self.inv = self.i.getInvertedIndex()
        self.di = self.i.getDirectIndex()
        self.doi = self.i.getDocumentIndex()
        self.doc_lengths = {}
        self.doc_n_unique_terms = {}

        # docno is used internally by terrier
        self.docno_to_docid = {}
        # doc_id is what we use
        self.docid_to_docno = {}

        log.info("Gathering document statistics")
        sum_doc_length = 0
        for docno in range(self.doi.getNumberOfDocuments()):
            doc_id = self.meta.getItem("docno", docno)
            doc = self.doi.getDocumentEntry(docno)
            doc_len = doc.getDocumentLength()
            sum_doc_length += doc_len
            self.doc_lengths[doc_id] = doc_len
            self.doc_n_unique_terms[doc_id] = doc.getNumberOfEntries()
            self.docno_to_docid[docno] = doc_id
            self.docno_to_docid[doc_id] = docno

        self.n_docs = len(self.doc_lengths)
        self.mean_doc_length = sum_doc_length / self.n_docs
        self.total_frequency = self.i.getCollectionStatistics().numberOfTokens

        log.info("Gathering document statistics complete!")

    def get_df(self, term):
        return self.lex[term].getDocumentFrequency() if term in self.lex else 0

    def get_docs_tf(self, term):
        le = self.lex.getLexiconEntry(term)
        # OOV
        if not le:
            return []
        doc_ids = []
        for posting in self.inv.getPostings(le):
            doc_id = self.docno_to_docid[posting.getId()]
            doc_ids.append((doc_id, posting.getFrequency()))
        return doc_ids

    def get_collection_tf(self, term):
        le = self.lex.getLexiconEntry(term)
        if not le:
            return 0
        return le.getFrequency()

    def get_doc_stats(self, doc_id):
        return {
            "n_unique_terms": self.doc_n_unique_terms[doc_id],
            "doc_len": self.doc_lengths[doc_id]
        }


class TerrierIndexer:
    def __init__(self, index_path, document_store, utils_inst, query_utils_inst, overwrite_index=False):
        self.index_path = index_path
        self.overwrite_index = overwrite_index
        self.utils = utils_inst
        self.query_utils = query_utils_inst

        documents = document_store.get_all_documents()
        if self.overwrite_index and os.path.exists(self.index_path):
            log.warning("Overwriting index")
            shutil.rmtree(self.index_path)

        if not os.path.exists(self.index_path):
            log.info("Building index")
            iter_indexer = pt.IterDictIndexer(self.index_path)
            iter_indexer.index(self.terrier_iter_dict(documents),
                               fields=("text",))
            log.info("Building index complete!")
        else:
            log.info("Using prebuilt index")

        self.index_ref = os.path.join(index_path, "data.properties")
        self.index_inst = pt.IndexFactory.of(self.index_ref)

        log.info(f"Index stats:\n{self.index_inst.getCollectionStatistics().toString()}")

    def get_index(self):
        return TerrierIndex(self.index_ref)

    def process(self, text):
        tokens = self.utils.tokenize(text)
        return " ".join(tokens)

    def process_query(self, text):
        tokens = self.query_utils.tokenize(text)
        return " ".join(tokens)

    def terrier_iter_dict(self, documents):
        N = len(documents)
        log.info(f"Iterating over {N} documents")
        print_every = max(N // 25, 5)
        for i, doc in enumerate(documents, 1):
            if i % print_every == 0:
                log.info(f"\t{i} of {N} done")
            yield {
                "text": self.process(doc.text),
                "docno": doc.id
            }


class TerrierRetriever(BaseRetriever):
    OVERWRITE_INDEX = False

    def __init__(self, document_store, config_json, top_k):
        super().__init__()
        self.top_k = top_k
        self.document_store = document_store
        self.documents = {}
        for doc in document_store.get_all_documents():
            self.documents[doc.id] = doc
        self.config = config_json
        # params passed on to PyTerrier
        self.controls = self.config["controls"]
        self.wmodel = self.config["wmodel"]
        log.info(f"WModel: {self.wmodel}")
        self.index_path = self.config["index_path"]
        self.index_ref = None
        self.meta_keys = None

        self.utils = get_std_utils()
        # Terrier expects cleaned data for queries only!
        self.query_utils = Utils(remove_square_braces=True, incl_only_alphanumeric=True)
        self.indexer = TerrierIndexer(self.index_path, document_store, self.utils, self.query_utils,
                                      self.OVERWRITE_INDEX)
        self.batch_retriever = pt.BatchRetrieve(self.indexer.index_inst, wmodel=self.wmodel, num_results=self.top_k,
                                                controls=self.controls)

    def batch_retrieve(self, queries: List[Tuple[str, str]], filters: dict = None,
                       index: str = None):
        if filters or index:
            raise NotImplementedError("filters/index not supported")

        topics = pd.DataFrame(queries, columns=['qid', 'query'])
        topics["query"] = topics["query"].apply(self.indexer.process_query)
        results = defaultdict(list)
        for _, row in self.batch_retriever.transform(topics).iterrows():
            results[row["qid"]].append((self.documents[row["docno"]], row["score"]))

        return results

    def retrieve(self, query, filters: dict = None, top_k: Optional[int] = None, index: str = None):
        if filters or index:
            raise NotImplementedError("filters/index not supported")

        if top_k:
            raise ValueError("Provide top_k arg only in constructor")

        results = []
        for _, row in self.batch_retriever.search(self.indexer.process_query(query.text)).iterrows():
            results.append((self.documents[row["docno"]], row["score"]))

        return results
