import unittest

from src.extract.append_all import AppendAll
from src.pipeline import read_pass
from src.retrieve.tfidf import Tfidf
from src.schema import Dialogue, QAItem, Reader, Turn


def _dialogue() -> Dialogue:
    return Dialogue(
        conv_id="conversation-1",
        speakers=["Alice", "Bob"],
        turns=[
            Turn(
                speaker="Alice",
                dia_id="D1:1",
                text="I adopted a tabby cat named Marmalade.",
                session_id="session_1",
                timestamp_raw="1 January 2024",
                timestamp="2024-01-01",
            ),
            Turn(
                speaker="Bob",
                dia_id="D1:2",
                text="My favorite sport is tennis.",
                session_id="session_1",
                timestamp_raw="1 January 2024",
                timestamp="2024-01-01",
            ),
        ],
    )


class FirstContextReader(Reader):
    def answer(self, question, context):
        return context[0].text if context else "no context"


class AppendAllTests(unittest.TestCase):
    def test_creates_one_memory_per_turn_with_provenance(self):
        memories = AppendAll().extract(_dialogue())

        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0].text, "Alice: I adopted a tabby cat named Marmalade.")
        self.assertEqual(memories[0].source_dia_ids, ["D1:1"])
        self.assertEqual(
            memories[0].meta,
            {
                "speaker": "Alice",
                "session_id": "session_1",
                "timestamp": "2024-01-01",
                "timestamp_raw": "1 January 2024",
            },
        )


class TfidfTests(unittest.TestCase):
    def setUp(self):
        self.memories = AppendAll().extract(_dialogue())
        self.retriever = Tfidf()
        self.retriever.index(self.memories)

    def test_returns_most_relevant_memory_first(self):
        results = self.retriever.retrieve("What is the cat's name?", k=2)

        self.assertEqual([memory.source_dia_ids for memory in results], [["D1:1"]])

    def test_handles_empty_index_nonpositive_k_and_unknown_terms(self):
        empty = Tfidf()
        empty.index([])

        self.assertEqual(empty.retrieve("cat", k=5), [])
        self.assertEqual(self.retriever.retrieve("cat", k=0), [])
        self.assertEqual(self.retriever.retrieve("completely-unseen-token", k=5), [])

    def test_reindex_replaces_previous_corpus(self):
        self.retriever.index(self.memories[:1])

        self.assertEqual(self.retriever.retrieve("tennis", k=1), [])


class PipelineSmokeTests(unittest.TestCase):
    def test_round_memory_baseline_runs_end_to_end(self):
        dialogue = _dialogue()
        qa = QAItem(
            conv_id=dialogue.conv_id,
            question="What is the cat's name?",
            gold_answer="Marmalade",
            evidence_dia_ids=["D1:1"],
            category=4,
        )

        records = read_pass(
            [dialogue],
            {dialogue.conv_id: [qa]},
            AppendAll(),
            Tfidf(),
            FirstContextReader(),
            k=1,
        )

        self.assertEqual(len(records), 1)
        self.assertIn("Marmalade", records[0][1])


if __name__ == "__main__":
    unittest.main()
