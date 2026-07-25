"""Runs the four stages, split into two passes.

read_pass runs extract, retrieve, and read using the reader.
judge_pass runs scoring using the judge.
Separating them keeps only one model resident at a time.
"""
from __future__ import annotations

from typing import Dict, List

from .schema import (
    Dialogue,
    Extractor,
    Judge,
    Prediction,
    QAItem,
    Reader,
    ReadRecord,
    Retriever,
)


def read_pass(
    dialogues: List[Dialogue],
    qa_by_conv: Dict[str, List[QAItem]],
    extractor: Extractor,
    retriever: Retriever,
    reader: Reader,
    k: int = 5,
) -> List[ReadRecord]:
    """Run extract, retrieve, and read while retaining retrieval provenance."""
    records: List[ReadRecord] = []
    for dlg in dialogues:
        memories = extractor.extract(dlg)
        retriever.index(memories)
        for qa in qa_by_conv[dlg.conv_id]:
            ctx = retriever.retrieve(qa.question, k)
            answer = reader.answer(qa.question, ctx)
            records.append(ReadRecord(
                qa_item=qa,
                answer_text=answer,
                retrieved_memories=ctx,
            ))
    return records


def judge_pass(records: List[ReadRecord], judge: Judge) -> List[Prediction]:
    """Score each qa, answer record into a Prediction."""
    preds: List[Prediction] = []
    for record in records:
        qa = record.qa_item
        score = judge.score(qa.question, record.answer_text, qa.gold_answer)
        preds.append(Prediction(
            qa_item=qa,
            answer_text=record.answer_text,
            judge_label=1 if score >= 0.5 else 0,
            judge_score=score,
            retrieved_memories=record.retrieved_memories,
        ))
    return preds
