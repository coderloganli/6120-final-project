from typing import List

from ..schema import Dialogue, Extractor, Memory
from .textfmt import memory_text


class AppendAll(Extractor):
    """Store every dialogue turn as an independent memory."""

    def __init__(self, with_timestamp: bool = False):
        self.with_timestamp = with_timestamp

    def extract(self, dialogue: Dialogue) -> List[Memory]:
        return [
            Memory(
                text=memory_text(turn, self.with_timestamp),
                source_dia_ids=[turn.dia_id],
                meta={
                    "speaker": turn.speaker,
                    "session_id": turn.session_id,
                    "timestamp": turn.timestamp,
                    "timestamp_raw": turn.timestamp_raw,
                },
            )
            for turn in dialogue.turns
        ]
