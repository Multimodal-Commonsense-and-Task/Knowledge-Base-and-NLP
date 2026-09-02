import json

from pyserini.encode import QueryEncoder as PyseriniQueryEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

from src.retriever.base import BaseRetriever
from src.util.dtype import Document, Query, RankedDocument

class ReasonIREncoder(PyseriniQueryEncoder):
    def __init__(self, dataset):
        self.encoder = SentenceTransformer('reasonir/ReasonIR-8B', trust_remote_code=True, model_kwargs={'torch_dtype': 'auto'})
        self.dataset = dataset

    def encode(self, query : str):
        embedding = self.encoder.encode([query], prompt=f'<|user|>\nGiven a {self.dataset} post, retrieve relevant passages that help answer the post\n<|embed|>\n', show_progress_bar=False)[0]
        return embedding

class ReasonIRRetriever(BaseRetriever):
    def __init__(self, dataset : str):
        self.encoder = ReasonIREncoder(dataset=dataset)
        self.searcher = FaissSearcher(f'index/{dataset}/reasonir', self.encoder)
        self.sparse_searcher = LuceneSearcher(f'index/{dataset}/bm25')

    def retrieve(self, query : Query, k : int = 10) -> list[RankedDocument]:
        search_result = self.searcher.search(query.text, k=k)
        if isinstance(search_result, list):
            hits = search_result
        else:
            _, hits = search_result
        results = []
        for hit in hits:
            did = hit.docid
            doc_obj = self.sparse_searcher.doc(hit.docid)
            doc_raw = doc_obj.raw() if doc_obj else ''
            text = json.loads(doc_raw).get('contents', '').strip()
            doc = Document(did=did, text=text)
            results.append(RankedDocument(document=doc, score=hit.score))
        return results
