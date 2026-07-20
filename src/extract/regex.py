from typing import List

from ..schema import Dialogue, Extractor, Memory


class Regex(Extractor):
    def extract(self, dialogue: Dialogue) -> List[Memory]:
        raise NotImplementedError
