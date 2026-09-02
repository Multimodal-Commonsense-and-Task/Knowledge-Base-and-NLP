import argparse

from alive_progress import alive_bar

from src.util.callback import Callback, DefaultRetrieverCallback
from src.util.const import register_method
from src.retriever import Retriever
from src.util.dtype import Query, RankedDocument

@register_method('reasonir')
class ReasonIR:
    def __init__(self, args: argparse.Namespace, tag: str, callbacks : list[Callback] | None = None):
        self.args = args
        self.retriever = Retriever(model_name='reasonir', dataset=args.dataset)
        if callbacks is None:
            self.callbacks: list[Callback] = [DefaultRetrieverCallback(dataset=args.dataset, query_type=args.query_type, file_name=f'{args.method}.{tag}.{args.idx}')]
        else:
            self.callbacks = callbacks

    def retrieve(self, query : Query, k : int = 10) -> list[RankedDocument]:
        ranks = self.retriever.retrieve(query=query, k=k + len(query.excluded_dids))
        ranks = [RankedDocument(document=rank.document, score=rank.score * 10000) for rank in ranks if rank.document.did not in query.excluded_dids][:k]
        for callback in self.callbacks:
            callback.on_query_end(query=query, ranks=ranks)
        return ranks

    def run(self, queries: list[Query], k : int = 10):
        with alive_bar(total=len(queries), title=f'Inference [{self.args.method.upper()}] on [{self.args.dataset}-{self.args.query_type}]') as bar:
            for query in queries:
                self.retrieve(query, k=k)
                bar()
        for callback in self.callbacks:
            callback.on_retrieval_end()
