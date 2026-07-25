import unittest

import numpy as np

from src.retrieve.bm25 import Bm25
from src.retrieve.sentence_emb import SentenceEmb
from src.retrieve.word2vec import Word2vec
from src.schema import Memory


def _memories():
    return [
        Memory(text="Alice: I adopted a tabby cat named Marmalade.", source_dia_ids=["D1:1"]),
        Memory(text="Bob: My favorite sport is tennis.", source_dia_ids=["D1:2"]),
    ]


def _toy_vectors():
    """Tiny word-vector table standing in for pretrained word2vec."""
    from gensim.models import KeyedVectors

    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ["cat", "kitten", "tennis", "sport"],
        np.array([
            [1.0, 0.0, 0.0],   # cat
            [0.9, 0.1, 0.0],   # kitten: close to cat
            [0.0, 1.0, 0.0],   # tennis
            [0.0, 0.9, 0.1],   # sport: close to tennis
        ]),
    )
    return kv


class Word2vecTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Word2vec(keyed_vectors=_toy_vectors())
        self.retriever.index(_memories())

    def test_ranks_semantically_close_memory_first(self):
        results = self.retriever.retrieve("kitten", k=1)

        self.assertEqual([m.source_dia_ids for m in results], [["D1:1"]])

    def test_handles_empty_index_nonpositive_k_and_unknown_terms(self):
        empty = Word2vec(keyed_vectors=_toy_vectors())
        empty.index([])

        self.assertEqual(empty.retrieve("cat", k=5), [])
        self.assertEqual(self.retriever.retrieve("cat", k=0), [])
        self.assertEqual(self.retriever.retrieve("completely-unseen-token", k=5), [])

    def test_reindex_replaces_previous_corpus(self):
        self.retriever.index(_memories()[:1])

        self.assertEqual(self.retriever.retrieve("tennis", k=1), [])


class Bm25Tests(unittest.TestCase):
    def setUp(self):
        self.retriever = Bm25()
        self.retriever.index(_memories())

    def test_returns_most_relevant_memory_first(self):
        results = self.retriever.retrieve("What is the cat's name?", k=2)

        self.assertEqual([m.source_dia_ids for m in results], [["D1:1"]])

    def test_handles_empty_index_nonpositive_k_and_unknown_terms(self):
        empty = Bm25()
        empty.index([])

        self.assertEqual(empty.retrieve("cat", k=5), [])
        self.assertEqual(self.retriever.retrieve("cat", k=0), [])
        self.assertEqual(self.retriever.retrieve("completely-unseen-token", k=5), [])

    def test_length_normalization_prefers_short_focused_memory(self):
        padding = " ".join(["filler"] * 30)
        memories = [
            Memory(text=f"cat {padding}", source_dia_ids=["D1:1"]),
            Memory(text="cat", source_dia_ids=["D1:2"]),
        ]
        self.retriever.index(memories)

        results = self.retriever.retrieve("cat", k=2)

        self.assertEqual(results[0].source_dia_ids, ["D1:2"])


class Word2vecWeightingTests(unittest.TestCase):
    """tfidf weighting should downweight corpus-frequent words."""

    def _corpus(self):
        return [
            Memory(text="cat", source_dia_ids=["D1:1"]),
            Memory(text="filler filler filler", source_dia_ids=["D1:2"]),
            Memory(text="tennis filler", source_dia_ids=["D1:3"]),
        ]

    def _vectors(self):
        from gensim.models import KeyedVectors

        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(
            ["cat", "kitten", "tennis", "filler"],
            np.array([
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )
        return kv

    def test_uniform_pooling_is_dominated_by_frequent_word(self):
        retriever = Word2vec(keyed_vectors=self._vectors(), weighting="uniform")
        retriever.index(self._corpus())

        results = retriever.retrieve("filler kitten", k=1)

        self.assertEqual(results[0].source_dia_ids, ["D1:2"])

    def test_tfidf_pooling_prefers_rare_content_word(self):
        retriever = Word2vec(keyed_vectors=self._vectors(), weighting="tfidf")
        retriever.index(self._corpus())

        results = retriever.retrieve("filler kitten", k=1)

        self.assertEqual(results[0].source_dia_ids, ["D1:1"])

    def test_rejects_unknown_weighting(self):
        with self.assertRaises(ValueError):
            Word2vec(keyed_vectors=self._vectors(), weighting="idf-only")


class Word2vecCaseFallbackTests(unittest.TestCase):
    def test_capitalized_vocabulary_is_reachable(self):
        from gensim.models import KeyedVectors

        kv = KeyedVectors(vector_size=2)
        # proper noun exists only capitalized, as in Google News vectors
        kv.add_vectors(["Boston", "city"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        retriever = Word2vec(keyed_vectors=kv)
        retriever.index([Memory(text="I moved to Boston.", source_dia_ids=["D1:1"]),
                         Memory(text="A big city.", source_dia_ids=["D1:2"])])

        results = retriever.retrieve("Boston", k=1)

        self.assertEqual(results[0].source_dia_ids, ["D1:1"])


class FakeEncoder:
    """Deterministic stand-in for a sentence-transformer: keyword indicator vectors."""

    _VOCAB = ["cat", "marmalade", "tennis", "sport"]

    def encode(self, texts, normalize_embeddings=False):
        rows = []
        for text in texts:
            lowered = text.lower()
            row = np.array([float(word in lowered) for word in self._VOCAB])
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.array(rows)


class RecordingEncoder(FakeEncoder):
    """FakeEncoder that also records every text passed to encode()."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings=False):
        self.seen.extend(texts)
        return super().encode(texts, normalize_embeddings=normalize_embeddings)


class PrefixPresetTests(unittest.TestCase):
    def test_symmetric_default_adds_no_prefixes(self):
        encoder = RecordingEncoder()
        retriever = SentenceEmb(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                encoder=encoder)
        retriever.index(_memories()[:1])
        retriever.retrieve("cat", k=1)

        self.assertTrue(encoder.seen[0].startswith("Alice:"))
        self.assertEqual(encoder.seen[1], "cat")

    def test_e5_preset_prefixes_query_and_passage(self):
        encoder = RecordingEncoder()
        retriever = SentenceEmb(model_name="intfloat/e5-small-v2", encoder=encoder)
        retriever.index(_memories()[:1])
        retriever.retrieve("cat", k=1)

        self.assertTrue(encoder.seen[0].startswith("passage: Alice:"))
        self.assertEqual(encoder.seen[1], "query: cat")

    def test_bge_preset_prefixes_query_only(self):
        encoder = RecordingEncoder()
        retriever = SentenceEmb(model_name="BAAI/bge-small-en-v1.5", encoder=encoder)
        retriever.index(_memories()[:1])
        retriever.retrieve("cat", k=1)

        self.assertTrue(encoder.seen[0].startswith("Alice:"))
        self.assertTrue(encoder.seen[1].startswith("Represent this sentence"))

    def test_explicit_prefix_overrides_preset(self):
        encoder = RecordingEncoder()
        retriever = SentenceEmb(model_name="intfloat/e5-small-v2", encoder=encoder,
                                query_prefix="", passage_prefix="")
        retriever.index(_memories()[:1])
        retriever.retrieve("cat", k=1)

        self.assertTrue(encoder.seen[0].startswith("Alice:"))
        self.assertEqual(encoder.seen[1], "cat")


class SentenceEmbTests(unittest.TestCase):
    def setUp(self):
        self.retriever = SentenceEmb(encoder=FakeEncoder())
        self.retriever.index(_memories())

    def test_ranks_relevant_memory_first(self):
        results = self.retriever.retrieve("What sport does Bob play, tennis?", k=1)

        self.assertEqual([m.source_dia_ids for m in results], [["D1:2"]])

    def test_handles_empty_index_and_nonpositive_k(self):
        empty = SentenceEmb(encoder=FakeEncoder())
        empty.index([])

        self.assertEqual(empty.retrieve("cat", k=5), [])
        self.assertEqual(self.retriever.retrieve("cat", k=0), [])


if __name__ == "__main__":
    unittest.main()
