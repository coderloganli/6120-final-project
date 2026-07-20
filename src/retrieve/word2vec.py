from typing import List

from ..schema import Memory, Retriever


class Word2vec(Retriever):
    def index(self, memories: List[Memory]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, k: int) -> List[Memory]:
        raise NotImplementedError
