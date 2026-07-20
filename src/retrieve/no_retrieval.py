from typing import List

from ..schema import Memory, Retriever


class NoRetrieval(Retriever):
    """Retrieves nothing. Pairs with NoMemory for the lower-bound baseline."""

    def index(self, memories: List[Memory]) -> None:
        pass

    def retrieve(self, query: str, k: int) -> List[Memory]:
        return []
