from typing import List

from ..schema import Dialogue, Extractor, Memory


class AppendAll(Extractor):
    """Store every dialogue turn as an independent memory."""

    def extract(self, dialogue: Dialogue) -> List[Memory]:
        return [
            Memory(
                text=f"{turn.speaker}: {turn.text}",
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
