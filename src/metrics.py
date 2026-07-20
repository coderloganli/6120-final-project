"""Aggregate Predictions into the comparison metric: overall QA accuracy."""
from typing import List

from .schema import Prediction


def summarize(preds: List[Prediction]) -> dict:
    n = len(preds)
    return {"n": n, "qa_accuracy": sum(p.judge_label for p in preds) / n if n else 0.0}
