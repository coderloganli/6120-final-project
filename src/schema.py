"""Data structures and the four stage interfaces.

Concrete methods subclass these interfaces. The loader, pipeline, and metrics
depend only on this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    speaker: str
    dia_id: str                     # LOCOMO turn id
    text: str
    session_id: str                 # LOCOMO session id
    timestamp_raw: str              # raw LOCOMO date string
    timestamp: Optional[str] = None # parsed ISO date, or None


@dataclass
class Dialogue:
    conv_id: str                    # LOCOMO sample_id
    speakers: List[str]
    turns: List[Turn]


@dataclass
class Memory:
    text: str
    source_dia_ids: List[str]       # source turn ids
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAItem:
    conv_id: str
    question: str
    gold_answer: str
    evidence_dia_ids: List[str]     # gold evidence turn ids
    category: int                   # 1=multi-hop 2=temporal 3=open-domain 4=single-hop


@dataclass
class ReadRecord:
    qa_item: QAItem
    answer_text: str
    retrieved_memories: List[Memory]


@dataclass
class Prediction:
    qa_item: QAItem
    answer_text: str
    judge_label: int                # 1 if correct, else 0
    judge_score: float = 0.0
    retrieved_memories: List[Memory] = field(default_factory=list)


class Extractor(ABC):
    @abstractmethod
    def extract(self, dialogue: Dialogue) -> List[Memory]:
        ...


class Retriever(ABC):
    @abstractmethod
    def index(self, memories: List[Memory]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Memory]:
        ...


class Reader(ABC):
    @abstractmethod
    def answer(self, question: str, context: List[Memory]) -> str:
        ...


class Judge(ABC):
    @abstractmethod
    def score(self, question: str, pred: str, gold: str) -> float:
        ...
