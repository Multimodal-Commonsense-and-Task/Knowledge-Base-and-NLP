import json

from pyserini.search.lucene import LuceneSearcher

from src.retriever.base import BaseRetriever
from src.util.dtype import Document, Query, RankedDocument

class BM25Retriever(BaseRetriever):
    def __init__(self, dataset : str, k1 : float = 0.9, b : float = 0.2):
        self.searcher = LuceneSearcher(f'index/{dataset}/bm25')
        self.searcher.set_bm25(k1=k1, b=b)

    def retrieve(self, query : Query, k : int = 10, excluded_dids : set[str] | None = None) -> list[RankedDocument]:
        if excluded_dids is None:
            excluded_dids = set()
        hits = self.searcher.search(query.text, k=k)
        results = []
        for hit in hits:
            did = hit.docid
            doc_obj = self.searcher.doc(hit.docid)
            doc_raw = doc_obj.raw() if doc_obj else ''
            text = json.loads(doc_raw).get('contents', '').strip()
            doc = Document(did=did, text=text)
            results.append(RankedDocument(document=doc, score=hit.score))
        return results
