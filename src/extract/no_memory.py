from typing import List

from ..schema import Dialogue, Extractor, Memory


class NoMemory(Extractor):
    """Lower-bound baseline. Stores no memory."""

    def extract(self, dialogue: Dialogue) -> List[Memory]:
        return []
