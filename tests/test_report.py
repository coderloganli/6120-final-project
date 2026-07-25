import json
import tempfile
import unittest
from pathlib import Path

from src.report import render_console, render_markdown, write_reports


def _metrics():
    group = {
        "n": 2,
        "qa_accuracy": 0.5,
        "judge_score": 0.5,
        "exact_match": 0.25,
        "token_f1": 0.75,
        "retrieval": {
            "n_with_evidence": 2,
            "evidence_hit_at_k": 1.0,
            "evidence_precision_at_k": 0.5,
            "evidence_recall_at_k": 0.75,
            "evidence_full_recall_at_k": 0.5,
            "mrr": 0.75,
            "avg_memories_retrieved": 2.0,
            "empty_retrieval_rate": 0.0,
        },
    }
    return {**group, "by_category": {"1": group}}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = [("append_all+tfidf", _metrics())]

    def test_renders_terminal_and_markdown_comparisons(self):
        console = render_console(self.rows, k=5)
        markdown = render_markdown(self.rows, k=5)

        self.assertIn("append_all+tfidf", console)
        self.assertIn("hit@5", console)
        self.assertIn("# Evaluation Report", markdown)
        self.assertIn("Evidence Recall@5", markdown)
        self.assertIn("1: Multi-hop", markdown)

    def test_writes_machine_and_human_readable_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_reports(self.rows, output_dir, k=5)

            payload = json.loads((output_dir / "summary.json").read_text())
            report = (output_dir / "report.md").read_text()

        self.assertEqual(payload["retrieval_k"], 5)
        self.assertEqual(payload["runs"][0]["combo"], "append_all+tfidf")
        self.assertIn("Overall comparison", report)


if __name__ == "__main__":
    unittest.main()
