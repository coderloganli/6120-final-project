"""Runs the four stages, split into two passes.

read_pass runs extract, retrieve, and read using the reader.
judge_pass runs scoring using the judge.
Separating them keeps only one model resident at a time.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .schema import Dialogue, Extractor, Judge, Prediction, QAItem, Reader, Retriever


def read_pass(
    dialogues: List[Dialogue],
    qa_by_conv: Dict[str, List[QAItem]],
    extractor: Extractor,
    retriever: Retriever,
    reader: Reader,
    k: int = 5,
) -> List[Tuple[QAItem, str]]:
    """Run extract, retrieve, and read. Return qa, answer records."""
    records: List[Tuple[QAItem, str]] = []
    for dlg in dialogues:
        memories = extractor.extract(dlg)
        retriever.index(memories)
        for qa in qa_by_conv[dlg.conv_id]:
            ctx = retriever.retrieve(qa.question, k)
            answer = reader.answer(qa.question, ctx)
            records.append((qa, answer))
    return records


def judge_pass(records: List[Tuple[QAItem, str]], judge: Judge) -> List[Prediction]:
    """Score each qa, answer record into a Prediction."""
    preds: List[Prediction] = []
    for qa, answer in records:
        score = judge.score(qa.question, answer, qa.gold_answer)
        preds.append(Prediction(qa_item=qa, answer_text=answer, judge_label=1 if score >= 0.5 else 0))
    return preds
