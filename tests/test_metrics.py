import unittest

from src.metrics import exact_match, normalize_answer, summarize, token_f1
from src.schema import Memory, Prediction, QAItem


def _prediction(
    *,
    category,
    gold,
    answer,
    correct,
    evidence,
    retrieved,
    judge_score=None,
):
    qa = QAItem(
        conv_id="conversation-1",
        question="Question?",
        gold_answer=gold,
        evidence_dia_ids=evidence,
        category=category,
    )
    memories = [
        Memory(text=dia_id, source_dia_ids=[dia_id]) for dia_id in retrieved
    ]
    return Prediction(
        qa_item=qa,
        answer_text=answer,
        judge_label=correct,
        judge_score=correct if judge_score is None else judge_score,
        retrieved_memories=memories,
    )


class AnswerMetricTests(unittest.TestCase):
    def test_normalized_exact_match_ignores_case_articles_and_punctuation(self):
        self.assertEqual(normalize_answer("The Marmalade!"), "marmalade")
        self.assertEqual(exact_match("The Marmalade!", "marmalade"), 1.0)

    def test_token_f1_gives_partial_credit(self):
        self.assertAlmostEqual(token_f1("orange tabby cat", "tabby cat"), 0.8)
        self.assertEqual(token_f1("Boston", "New York"), 0.0)


class SummaryTests(unittest.TestCase):
    def test_summarizes_answer_retrieval_and_category_metrics(self):
        predictions = [
            _prediction(
                category=1,
                gold="Marmalade",
                answer="The Marmalade.",
                correct=1,
                evidence=["D1:1", "D1:2"],
                retrieved=["D1:1", "D1:9"],
            ),
            _prediction(
                category=2,
                gold="New York",
                answer="Boston",
                correct=0,
                evidence=["D2:2"],
                retrieved=["D2:8", "D2:2"],
            ),
        ]

        metrics = summarize(predictions)

        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["qa_accuracy"], 0.5)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["token_f1"], 0.5)
        self.assertEqual(metrics["retrieval"]["n_with_evidence"], 2)
        self.assertEqual(metrics["retrieval"]["evidence_hit_at_k"], 1.0)
        self.assertEqual(metrics["retrieval"]["evidence_precision_at_k"], 0.5)
        self.assertEqual(metrics["retrieval"]["evidence_recall_at_k"], 0.75)
        self.assertEqual(metrics["retrieval"]["evidence_full_recall_at_k"], 0.5)
        self.assertEqual(metrics["retrieval"]["mrr"], 0.75)
        self.assertEqual(metrics["avg_memories_retrieved"], 2)
        self.assertEqual(metrics["empty_retrieval_rate"], 0.0)
        self.assertEqual(set(metrics["by_category"]), {"1", "2"})
        self.assertEqual(metrics["by_category"]["1"]["qa_accuracy"], 1.0)

    def test_excludes_missing_gold_evidence_from_evidence_metrics(self):
        prediction = _prediction(
            category=3,
            gold="Answer",
            answer="Answer",
            correct=1,
            evidence=[],
            retrieved=[],
        )

        metrics = summarize([prediction])

        self.assertEqual(metrics["retrieval"]["n_with_evidence"], 0)
        self.assertEqual(metrics["retrieval"]["evidence_hit_at_k"], 0.0)
        self.assertEqual(metrics["empty_retrieval_rate"], 1.0)

    def test_empty_predictions_are_safe(self):
        metrics = summarize([])

        self.assertEqual(metrics["n"], 0)
        self.assertEqual(metrics["qa_accuracy"], 0.0)
        self.assertEqual(metrics["by_category"], {})


if __name__ == "__main__":
    unittest.main()
