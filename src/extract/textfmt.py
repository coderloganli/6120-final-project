"""Shared memory-text formatting for extractors.

with_timestamp=True prepends the session date, giving the reader an anchor to
resolve relative time expressions ("yesterday", "last year") into absolute
dates — LOCOMO temporal gold answers are absolute. Falls back to the raw
LOCOMO date string when the parsed ISO date is unavailable.
"""
from ..schema import Turn


def memory_text(turn: Turn, with_timestamp: bool = False) -> str:
    base = f"{turn.speaker}: {turn.text}"
    if not with_timestamp:
        return base
    stamp = turn.timestamp or turn.timestamp_raw
    return f"[{stamp}] {base}" if stamp else base
