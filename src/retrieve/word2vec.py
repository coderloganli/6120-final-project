"""Static-embedding retrieval: average word2vec vectors, rank by cosine.

Each memory is embedded as a pooled vector of its word vectors; the query is
embedded the same way. Tokenization mirrors the TF-IDF tokenizer (same word
regex and stop words) so the sparse and dense retrievers see the same tokens,
but case is preserved for vector lookup: the Google News vocabulary is
case-sensitive and proper nouns ("Boston") often exist only capitalized.
Lookup tries the original form first, then the lowercased form.

Two pooling modes:
- weighting="uniform": plain mean of word vectors
- weighting="tfidf": words weighted by term-count x IDF (IDF fitted on the
  indexed corpus with the same smoothing as the TF-IDF retriever), so frequent
  chit-chat words contribute less than rare content words

The default model is the pretrained Google News word2vec (Mikolov et al.,
2013). First use downloads ~1.6GB via gensim; override with W2V_MODEL, e.g.
W2V_MODEL=glove-wiki-gigaword-100 for a 128MB development model.
"""
import math
import os
from collections import Counter
from typing import List, Tuple

import numpy as np

from ..schema import Memory, Retriever
from .tfidf import _STOP_WORDS, _TOKEN

# Default model. Override with W2V_MODEL for smaller downloads.
MODEL = os.environ.get("W2V_MODEL", "word2vec-google-news-300")

WEIGHTINGS = ("uniform", "tfidf")


def _case_tokens(text: str) -> List[Tuple[str, str]]:
    """Return (original, lowercased) token pairs, stop words removed."""
    pairs = [(w, w.lower()) for w in _TOKEN.findall(text)]
    return [(w, lw) for w, lw in pairs if lw not in _STOP_WORDS]


class Word2vec(Retriever):
    """Rank memories by cosine similarity of pooled word vectors."""

    def __init__(self, model_name: str = MODEL, keyed_vectors=None,
                 weighting: str = "uniform"):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        if keyed_vectors is not None:  # injectable for tests
            self.vectors = keyed_vectors
        else:
            import gensim.downloader  # lazy import, mirrors LocalLLMReader
            self.vectors = gensim.downloader.load(model_name)
        self.weighting = weighting
        self._idf = {}
        self._default_idf = 1.0
        self._memories: List[Memory] = []
        self._matrix: np.ndarray = np.zeros((0, self.vectors.vector_size))

    def _lookup(self, word: str, lowered: str):
        if word in self.vectors:
            return self.vectors[word]
        if lowered in self.vectors:
            return self.vectors[lowered]
        return None

    def _embed(self, text: str) -> np.ndarray:
        pairs = _case_tokens(text)
        counts = Counter(lw for _, lw in pairs)
        vecs, weights = [], []
        for word, lowered in dict(pairs).items():  # each surface form once
            vec = self._lookup(word, lowered)
            if vec is None:
                continue
            if self.weighting == "tfidf":
                weight = counts[lowered] * self._idf.get(lowered, self._default_idf)
            else:
                weight = counts[lowered]
            vecs.append(vec)
            weights.append(weight)
        if not vecs:
            return np.zeros(self.vectors.vector_size)
        return np.average(vecs, axis=0, weights=weights)

    def index(self, memories: List[Memory]) -> None:
        self._memories = list(memories)
        if self.weighting == "tfidf":
            documents = [{lw for _, lw in _case_tokens(m.text)} for m in self._memories]
            n_documents = len(documents)
            document_frequency = Counter(term for doc in documents for term in doc)
            # Same smoothing as the TF-IDF retriever.
            self._idf = {
                term: math.log((1 + n_documents) / (1 + frequency)) + 1
                for term, frequency in document_frequency.items()
            }
            self._default_idf = math.log(1 + n_documents) + 1
        if not self._memories:
            self._matrix = np.zeros((0, self.vectors.vector_size))
            return
        self._matrix = np.stack([self._embed(m.text) for m in self._memories])

    def retrieve(self, query: str, k: int) -> List[Memory]:
        if k <= 0 or not self._memories:
            return []
        q = self._embed(query)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        norms = np.linalg.norm(self._matrix, axis=1)
        valid = norms > 0
        scores = np.zeros(len(self._memories))
        scores[valid] = (self._matrix[valid] @ q) / (norms[valid] * q_norm)
        top = np.argsort(-scores, kind="stable")[:k]
        return [self._memories[i] for i in top if scores[i] > 0]
