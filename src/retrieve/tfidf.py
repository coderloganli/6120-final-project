"""A small dependency-free TF-IDF cosine retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from ..schema import Memory, Retriever

_TOKEN = re.compile(r"\b\w+\b", flags=re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "s",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokens(text: str) -> List[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP_WORDS]


def _tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    if not counts:
        return {}
    return {term: count * idf[term] for term, count in counts.items() if term in idf}


def _norm(vector: Dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


class Tfidf(Retriever):
    """Rank memories by cosine similarity in a fitted TF-IDF space."""

    def __init__(self) -> None:
        self._memories: List[Memory] = []
        self._idf: Dict[str, float] = {}
        self._vectors: List[Dict[str, float]] = []
        self._norms: List[float] = []

    def index(self, memories: List[Memory]) -> None:
        self._memories = list(memories)
        documents = [_tokens(memory.text) for memory in self._memories]
        n_documents = len(documents)
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        # Smooth both numerator and denominator so unseen/small corpora behave well.
        self._idf = {
            term: math.log((1 + n_documents) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [_tfidf(document, self._idf) for document in documents]
        self._norms = [_norm(vector) for vector in self._vectors]

    def retrieve(self, query: str, k: int) -> List[Memory]:
        if k <= 0 or not self._memories:
            return []

        query_vector = _tfidf(_tokens(query), self._idf)
        query_norm = _norm(query_vector)
        if query_norm == 0:
            return []

        scored = []
        for index, (vector, vector_norm) in enumerate(zip(self._vectors, self._norms)):
            if vector_norm == 0:
                continue
            dot = sum(
                query_weight * vector.get(term, 0.0)
                for term, query_weight in query_vector.items()
            )
            score = dot / (query_norm * vector_norm)
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._memories[index] for _, index in scored[:k]]
