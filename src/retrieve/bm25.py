"""BM25 retrieval: the standard upgrade over raw TF-IDF cosine.

Same tokenizer and index/retrieve contract as the TF-IDF retriever; only the
scoring changes (Robertson & Zaragoza, 2009):

- term-frequency saturation: repeating a word in one memory stops helping
  after a few occurrences (k1 controls how fast it saturates)
- length normalization: long memories are penalized so they cannot win just
  by containing more words (b controls the strength)

k1=1.5, b=0.75 are the conventional defaults.
"""
import math
from collections import Counter
from typing import Dict, List

from ..schema import Memory, Retriever
from .tfidf import _tokens

K1 = 1.5
B = 0.75


class Bm25(Retriever):
    """Rank memories by BM25 score."""

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self._memories: List[Memory] = []
        self._doc_counts: List[Counter] = []
        self._doc_lens: List[int] = []
        self._avg_len = 0.0
        self._idf: Dict[str, float] = {}

    def index(self, memories: List[Memory]) -> None:
        self._memories = list(memories)
        documents = [_tokens(m.text) for m in self._memories]
        self._doc_counts = [Counter(doc) for doc in documents]
        self._doc_lens = [len(doc) for doc in documents]
        n_documents = len(documents)
        self._avg_len = (sum(self._doc_lens) / n_documents) if n_documents else 0.0
        document_frequency = Counter(term for doc in documents for term in set(doc))
        # BM25+-style floor keeps idf positive for very common terms.
        self._idf = {
            term: math.log(1 + (n_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, k: int) -> List[Memory]:
        if k <= 0 or not self._memories or self._avg_len == 0:
            return []
        query_terms = [t for t in _tokens(query) if t in self._idf]
        if not query_terms:
            return []
        scored = []
        for index, (counts, length) in enumerate(zip(self._doc_counts, self._doc_lens)):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._memories[index] for _, index in scored[:k]]
