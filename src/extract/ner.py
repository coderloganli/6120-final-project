"""NER-based extraction: keep turns that mention at least one salient entity.

Runs spaCy NER over each turn and stores the turn as one memory when it
contains an entity of a salient type (people, places, organizations, dates,
events, ...). Turns with no entities are treated as chit-chat and dropped.
Memory text keeps the same "speaker: text" format as AppendAll so retrievers
see a comparable corpus; recognized entities are kept in meta for analysis.

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""
import os
from typing import List

from ..schema import Dialogue, Extractor, Memory

# Default model. Override with NER_MODEL for a larger model.
MODEL = os.environ.get("NER_MODEL", "en_core_web_sm")

SALIENT_LABELS = {
    "PERSON", "GPE", "LOC", "ORG", "NORP", "FAC",
    "EVENT", "WORK_OF_ART", "PRODUCT", "DATE", "TIME",
}


class NER(Extractor):
    """Keep turns containing at least one salient named entity."""

    def __init__(self, model_name: str = MODEL):
        import spacy  # lazy import, mirrors LocalLLMReader
        self.nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "attribute_ruler"])

    def extract(self, dialogue: Dialogue) -> List[Memory]:
        memories = []
        docs = self.nlp.pipe(turn.text for turn in dialogue.turns)
        for turn, doc in zip(dialogue.turns, docs):
            entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in SALIENT_LABELS]
            if not entities:
                continue
            memories.append(Memory(
                text=f"{turn.speaker}: {turn.text}",
                source_dia_ids=[turn.dia_id],
                meta={
                    "speaker": turn.speaker,
                    "session_id": turn.session_id,
                    "timestamp": turn.timestamp,
                    "timestamp_raw": turn.timestamp_raw,
                    "entities": entities,
                },
            ))
        return memories
