"""Contextual-embedding retrieval: sentence-transformers, rank by cosine.

Each memory is encoded as one dense vector by a pretrained sentence encoder;
the query is encoded the same way. Unlike word2vec averaging, the encoder
sees word order and context, so paraphrases score high without token overlap.

The default model is all-MiniLM-L6-v2 (~80MB, symmetric encoder). Override
with SENT_MODEL. Known retrieval-specialized (asymmetric) alternatives and
their required text prefixes are registered in PREFIX_PRESETS and applied
automatically:

- multi-qa-MiniLM-L6-cos-v1: same size/speed as MiniLM but trained on QA
  pairs (no prefixes needed)
- BAAI/bge-small-en-v1.5: query-side instruction prefix
- intfloat/e5-small-v2: "query: " / "passage: " prefixes on both sides
"""
import os
from typing import List

import numpy as np

from ..schema import Memory, Retriever

# Default model. Override with SENT_MODEL.
MODEL = os.environ.get("SENT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# (query_prefix, passage_prefix) conventions required by each model family.
# Matched by substring so hub-prefixed names like "intfloat/e5-small-v2" hit.
PREFIX_PRESETS = {
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
    "e5-": ("query: ", "passage: "),
}


def _prefixes_for(model_name: str):
    lowered = model_name.lower()
    for marker, prefixes in PREFIX_PRESETS.items():
        if marker in lowered:
            return prefixes
    return ("", "")


class SentenceEmb(Retriever):
    """Rank memories by cosine similarity of sentence embeddings."""

    def __init__(self, model_name: str = MODEL, encoder=None,
                 query_prefix: str = None, passage_prefix: str = None):
        if encoder is not None:  # injectable for tests
            self.encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer  # lazy import
            self.encoder = SentenceTransformer(model_name)
        preset_query, preset_passage = _prefixes_for(model_name)
        self.query_prefix = preset_query if query_prefix is None else query_prefix
        self.passage_prefix = preset_passage if passage_prefix is None else passage_prefix
        self._memories: List[Memory] = []
        self._matrix: np.ndarray = np.zeros((0, 0))

    def index(self, memories: List[Memory]) -> None:
        self._memories = list(memories)
        if not self._memories:
            self._matrix = np.zeros((0, 0))
            return
        # normalize_embeddings=True makes dot product equal cosine similarity
        self._matrix = np.asarray(self.encoder.encode(
            [self.passage_prefix + m.text for m in self._memories],
            normalize_embeddings=True,
        ))

    def retrieve(self, query: str, k: int) -> List[Memory]:
        if k <= 0 or not self._memories:
            return []
        q = np.asarray(self.encoder.encode(
            [self.query_prefix + query], normalize_embeddings=True,
        ))[0]
        scores = self._matrix @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [self._memories[i] for i in top]
