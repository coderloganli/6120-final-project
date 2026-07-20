"""Load LOCOMO into Dialogue and QAItem objects.

Handles the data quirks: answers may be int and are coerced to str; category 5
has no answer and is excluded; the conversation is split across session_N lists
with session_N_date_time strings; evidence strings are normalized to Dn:m ids.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.schema import Dialogue, QAItem, Turn

DEFAULT_PATH = Path(__file__).parent / "locomo10.json"
EXCLUDED_CATEGORIES = {5}

_DIA_ID = re.compile(r"D:?(\d+):(\d+)")


def _parse_timestamp(raw: str):
    """Parse a LOCOMO date string to an ISO date. Return None on failure."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw or "")
    if not m:
        return None
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_evidence(raw) -> List[str]:
    """Extract well-formed Dn:m ids from evidence entries; drop leading zeros."""
    ids = []
    for entry in raw:
        for sess, turn in _DIA_ID.findall(str(entry)):
            ids.append(f"D{int(sess)}:{int(turn)}")
    return ids


def _load_dialogue(sample: dict) -> Dialogue:
    conv = sample["conversation"]
    speakers = [conv.get("speaker_a", ""), conv.get("speaker_b", "")]
    session_ids = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)),
        key=lambda s: int(s.split("_")[1]),
    )
    turns: List[Turn] = []
    for sid in session_ids:
        raw_ts = conv.get(f"{sid}_date_time", "")
        parsed = _parse_timestamp(raw_ts)
        for t in conv[sid]:
            turns.append(Turn(
                speaker=t["speaker"], dia_id=t["dia_id"], text=t["text"],
                session_id=sid, timestamp_raw=raw_ts, timestamp=parsed,
            ))
    return Dialogue(conv_id=sample["sample_id"], speakers=speakers, turns=turns)


def _load_qa(sample: dict) -> List[QAItem]:
    items = []
    for q in sample["qa"]:
        if q.get("category") in EXCLUDED_CATEGORIES:
            continue
        items.append(QAItem(
            conv_id=sample["sample_id"],
            question=q["question"],
            gold_answer=str(q.get("answer", "")),
            evidence_dia_ids=_normalize_evidence(q.get("evidence", [])),
            category=q["category"],
        ))
    return items


def load_locomo(path=DEFAULT_PATH) -> Tuple[List[Dialogue], Dict[str, List[QAItem]]]:
    """Return (dialogues, qa_by_conv_id)."""
    data = json.load(open(path))
    dialogues = [_load_dialogue(s) for s in data]
    qa_by_conv = {s["sample_id"]: _load_qa(s) for s in data}
    return dialogues, qa_by_conv


if __name__ == "__main__":
    dlgs, qa = load_locomo()
    n_qa = sum(len(v) for v in qa.values())
    print(f"loaded {len(dlgs)} dialogues, {n_qa} QA items (cat5 excluded)")
