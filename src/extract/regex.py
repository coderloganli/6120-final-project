"""Rule-based extraction: keep turns that match hand-written information patterns.

The design adapts the persona-extraction heuristic of Mazare et al. (2018),
"Training Millions of Personalized Dialogue Agents" (EMNLP, arXiv:1809.01984),
which mined 5M persona sentences from Reddit by keeping sentences that contain
"I" or "my" plus a verb filter. We refine that I/my heuristic into six named
pattern groups so each kind of personal fact can be analyzed separately:

- past_event / state / preference: split Mazare's "I + verb" rule by verb class
  (first-person events, states/identity, likes and plans)
- possessive: Mazare's "my + noun" rule
- temporal / plan: additions beyond Mazare, targeting LOCOMO's temporal QA
  category (time expressions) and future intentions

The pattern categories also mirror the personal-attribute taxonomy (hobbies,
family, possessions, likes/dislikes) of Wu et al. (2022), "Extracting and
Inferring Personal Attributes from Dialogue" (arXiv:2109.12702).

A turn is stored as one memory when any pattern fires; chit-chat turns match
nothing and are dropped. Memory text keeps the same "speaker: text" format as
AppendAll so retrievers see a comparable corpus.
"""
import re
from typing import List

from ..schema import Dialogue, Extractor, Memory
from .textfmt import memory_text

PATTERNS = {
    # "I went to a support group yesterday", "I ran a charity race"
    "past_event": re.compile(
        r"\bI\s+(?:went|got|ran|made|took|saw|met|had|did|bought|sold|started|"
        r"finished|joined|visited|tried|found|won|lost|moved|adopted|read|wrote|"
        r"quit|left|began|became|broke|spoke|taught|thought|felt|kept|held|"
        r"\w+ed)\b",
        re.IGNORECASE,
    ),
    # "I'm keen on counseling", "I was a teacher"
    "state": re.compile(r"\bI\s*(?:'m|am|was)\b", re.IGNORECASE),
    # "I love hiking", "I want to be a counselor"
    "preference": re.compile(
        r"\bI\s+(?:love|like|enjoy|hate|dislike|prefer|want|hope|plan|need|wish)\b",
        re.IGNORECASE,
    ),
    # "my kids", "my favorite sport", "my new job"
    "possessive": re.compile(r"\bmy\s+\w+", re.IGNORECASE),
    # "yesterday", "last Saturday", "two years ago"
    "temporal": re.compile(
        r"\b(?:yesterday|today|tomorrow|tonight|last\s+\w+|next\s+\w+|"
        r"\w+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
        re.IGNORECASE,
    ),
    # "gonna continue my edu", "we're thinking about going camping"
    "plan": re.compile(r"\b(?:gonna|going\s+to|planning\s+(?:to|on)|plan\s+to)\b", re.IGNORECASE),
}


# v2: data-driven revision after per-pattern analysis on LOCOMO gold evidence.
# Changes, each backed by the analysis:
# - "we" support everywhere: missed-evidence review showed shared activities
#   ("We made pizza!") are consistently phrased with we/our, which v1 skipped
# - contraction support: v1's "\bI\s+" never matched "I've worked" because of
#   the apostrophe, losing perfect-tense facts
# - state tightened: v1 state had the worst volume/precision ratio (17.8% of
#   turns at 32.6% precision); v2 excludes discourse fillers ("I'm so glad",
#   "I'm off to ...") via a lookahead
PATTERNS_V2 = {
    "past_event": re.compile(
        r"\b(?:I|we)(?:'ve)?\s+(?:went|got|ran|made|took|saw|met|had|did|bought|"
        r"sold|started|finished|joined|visited|tried|found|won|lost|moved|adopted|"
        r"read|wrote|quit|left|began|became|broke|spoke|taught|thought|felt|kept|"
        r"held|been|done|\w+ed)\b",
        re.IGNORECASE,
    ),
    "state": re.compile(
        r"\b(?:I|we)\s*(?:'m|'re|am|are|was|were)\b"
        r"(?!\s+(?:so\s+|really\s+|very\s+)?"
        r"(?:glad|happy|sorry|sure|excited|proud of you|off\b|gonna|going to))",
        re.IGNORECASE,
    ),
    "preference": re.compile(
        r"\b(?:I|we)(?:'d)?\s+(?:love|like|enjoy|hate|dislike|prefer|want|hope|"
        r"plan|need|wish)\b",
        re.IGNORECASE,
    ),
    "possessive": re.compile(r"\b(?:my|our)\s+\w+", re.IGNORECASE),
    "temporal": PATTERNS["temporal"],
    "plan": PATTERNS["plan"],
}

VERSIONS = {"v1": PATTERNS, "v2": PATTERNS_V2}


class Regex(Extractor):
    """Keep turns matching at least one information-bearing pattern."""

    def __init__(self, version: str = "v1", with_timestamp: bool = False):
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {tuple(VERSIONS)}, got {version!r}")
        self.patterns = VERSIONS[version]
        self.with_timestamp = with_timestamp

    def extract(self, dialogue: Dialogue) -> List[Memory]:
        memories = []
        for turn in dialogue.turns:
            matched = [name for name, pattern in self.patterns.items() if pattern.search(turn.text)]
            if not matched:
                continue
            memories.append(Memory(
                text=memory_text(turn, self.with_timestamp),
                source_dia_ids=[turn.dia_id],
                meta={
                    "speaker": turn.speaker,
                    "session_id": turn.session_id,
                    "timestamp": turn.timestamp,
                    "timestamp_raw": turn.timestamp_raw,
                    "matched_patterns": matched,
                },
            ))
        return memories
