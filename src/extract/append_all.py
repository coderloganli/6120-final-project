from typing import List

from ..schema import Dialogue, Extractor, Memory


class AppendAll(Extractor):
    def extract(self, dialogue: Dialogue) -> List[Memory]:
        raise NotImplementedError
