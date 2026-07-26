"""Runs the four stages, split into two passes.

read_pass runs extract, retrieve, and read using the reader.
judge_pass runs scoring using the judge.
Separating them keeps only one model resident at a time. Both passes call the
model in batches of batch_size to use the GPU efficiently.
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


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def read_pass(
    dialogues: List[Dialogue],
    qa_by_conv: Dict[str, List[QAItem]],
    extractor: Extractor,
    retriever: Retriever,
    reader: Reader,
    k: int = 5,
    batch_size: int = 16,
) -> List[ReadRecord]:
    """Extract and retrieve for every question, then read the answers in batches."""
    # Phase A: extract + retrieve (cheap, no model). Collect (qa, context) pairs.
    pending = []
    for dlg in dialogues:
        memories = extractor.extract(dlg)
        retriever.index(memories)
        for qa in qa_by_conv[dlg.conv_id]:
            pending.append((qa, retriever.retrieve(qa.question, k)))

    # Phase B: reader answers in batches.
    records: List[ReadRecord] = []
    for chunk in _chunks(pending, batch_size):
        answers = reader.answer_batch([qa.question for qa, _ in chunk], [ctx for _, ctx in chunk])
        for (qa, ctx), answer in zip(chunk, answers):
            records.append(ReadRecord(qa_item=qa, answer_text=answer, retrieved_memories=ctx))
    return records


def judge_pass(records: List[ReadRecord], judge: Judge, batch_size: int = 16) -> List[Prediction]:
    """Score each answer against its gold answer, in batches."""
    preds: List[Prediction] = []
    for chunk in _chunks(records, batch_size):
        triples = [(r.qa_item.question, r.answer_text, r.qa_item.gold_answer) for r in chunk]
        scores = judge.score_batch(triples)
        for record, score in zip(chunk, scores):
            preds.append(Prediction(
                qa_item=record.qa_item,
                answer_text=record.answer_text,
                judge_label=1 if score >= 0.5 else 0,
                judge_score=score,
                retrieved_memories=record.retrieved_memories,
            ))
    return preds
