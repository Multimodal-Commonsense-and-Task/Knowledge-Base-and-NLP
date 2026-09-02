from abc import ABC, abstractmethod

from src.util.dtype import Query, RankedDocument


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: Query, k: int = 10) -> list[RankedDocument]:
        raise NotImplementedError
