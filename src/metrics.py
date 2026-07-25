"""Answer and retrieval metrics for memory-augmented QA experiments."""

from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List

from .schema import Memory, Prediction

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Apply the standard lowercase/punctuation/article normalization."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def source_dia_ids(memories: Iterable[Memory]) -> List[str]:
    """Flatten retrieved source ids in rank order, removing duplicates."""
    seen = set()
    result = []
    for memory in memories:
        for dia_id in memory.source_dia_ids:
            if dia_id not in seen:
                seen.add(dia_id)
                result.append(dia_id)
    return result


def _retrieval_scores(prediction: Prediction) -> Dict[str, float]:
    gold = set(prediction.qa_item.evidence_dia_ids)
    retrieved = source_dia_ids(prediction.retrieved_memories)
    retrieved_set = set(retrieved)
    overlap = gold & retrieved_set

    first_relevant_rank = next(
        (
            rank
            for rank, memory in enumerate(prediction.retrieved_memories, start=1)
            if gold.intersection(memory.source_dia_ids)
        ),
        None,
    )
    return {
        "evidence_hit_at_k": float(bool(overlap)),
        "evidence_precision_at_k": len(overlap) / len(retrieved_set) if retrieved_set else 0.0,
        "evidence_recall_at_k": len(overlap) / len(gold),
        "evidence_full_recall_at_k": float(gold <= retrieved_set),
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def _summarize_group(preds: List[Prediction]) -> dict:
    n = len(preds)
    evidence_preds = [pred for pred in preds if pred.qa_item.evidence_dia_ids]
    retrieval = [_retrieval_scores(pred) for pred in evidence_preds]

    def average(values):
        return mean(values) if values else 0.0

    return {
        "n": n,
        "qa_accuracy": average([pred.judge_label for pred in preds]),
        "judge_score": average([pred.judge_score for pred in preds]),
        "exact_match": average([
            exact_match(pred.answer_text, pred.qa_item.gold_answer) for pred in preds
        ]),
        "token_f1": average([
            token_f1(pred.answer_text, pred.qa_item.gold_answer) for pred in preds
        ]),
        "retrieval": {
            "n_with_evidence": len(evidence_preds),
            "evidence_hit_at_k": average([
                scores["evidence_hit_at_k"] for scores in retrieval
            ]),
            "evidence_precision_at_k": average([
                scores["evidence_precision_at_k"] for scores in retrieval
            ]),
            "evidence_recall_at_k": average([
                scores["evidence_recall_at_k"] for scores in retrieval
            ]),
            "evidence_full_recall_at_k": average([
                scores["evidence_full_recall_at_k"] for scores in retrieval
            ]),
            "mrr": average([scores["mrr"] for scores in retrieval]),
            "avg_memories_retrieved": average([
                len(pred.retrieved_memories) for pred in preds
            ]),
            "empty_retrieval_rate": average([
                not pred.retrieved_memories for pred in preds
            ]),
        },
    }


def summarize(preds: List[Prediction]) -> dict:
    """Return overall metrics plus a LOCOMO category breakdown."""
    result = _summarize_group(preds)
    categories = sorted({pred.qa_item.category for pred in preds})
    result["by_category"] = {
        str(category): _summarize_group([
            pred for pred in preds if pred.qa_item.category == category
        ])
        for category in categories
    }
    return result
