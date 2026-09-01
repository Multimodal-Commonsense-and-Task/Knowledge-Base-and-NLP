from collections.abc import Callable

from src.retriever.base import BaseRetriever
from src.retriever.bm25 import BM25Retriever
from src.retriever.reasonir import ReasonIRRetriever
from src.util.dtype import Query, RankedDocument


class Retriever:
    MODEL_MAPPING: dict[str, Callable[..., BaseRetriever]] = {
        'bm25': BM25Retriever,
        'reasonir': ReasonIRRetriever,
    }

    def __init__(self, model_name: str, **model_kwargs):
        try:
            retriever_factory = self.MODEL_MAPPING[model_name]
        except KeyError as ex:
            raise ValueError(f'Unsupported retriever model name: [{model_name}]') from ex

        try:
            self.retriever: BaseRetriever = retriever_factory(**model_kwargs)
        except TypeError as ex:
            raise TypeError(f'Unable to instantiate retriever [{model_name}] with arguments: {model_kwargs}') from ex

    def retrieve(self, query: Query, k: int = 10) -> list[RankedDocument]:
        return self.retriever.retrieve(query=query, k=k)
