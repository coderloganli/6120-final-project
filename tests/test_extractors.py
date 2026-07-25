import unittest

from src.extract.append_all import AppendAll
from src.extract.regex import Regex
from src.schema import Dialogue, Turn


def _turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(
        speaker=speaker,
        dia_id=dia_id,
        text=text,
        session_id="session_1",
        timestamp_raw="1 January 2024",
        timestamp="2024-01-01",
    )


def _dialogue(*turns: Turn) -> Dialogue:
    return Dialogue(conv_id="conversation-1", speakers=["Alice", "Bob"], turns=list(turns))


class RegexTests(unittest.TestCase):
    def test_keeps_fact_turns_and_drops_chitchat(self):
        dialogue = _dialogue(
            _turn("D1:1", "Alice", "Hey Bob! Good to see you!"),
            _turn("D1:2", "Bob", "I adopted a tabby cat named Marmalade last week."),
            _turn("D1:3", "Alice", "Wow, that's cool!"),
        )

        memories = Regex().extract(dialogue)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].text, "Bob: I adopted a tabby cat named Marmalade last week.")
        self.assertEqual(memories[0].source_dia_ids, ["D1:2"])

    def test_meta_records_which_patterns_matched(self):
        dialogue = _dialogue(
            _turn("D1:1", "Alice", "I went to a support group yesterday."),
        )

        memories = Regex().extract(dialogue)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].meta["speaker"], "Alice")
        self.assertEqual(memories[0].meta["timestamp"], "2024-01-01")
        self.assertIn("past_event", memories[0].meta["matched_patterns"])
        self.assertIn("temporal", memories[0].meta["matched_patterns"])

    def test_preference_and_possessive_patterns(self):
        dialogue = _dialogue(
            _turn("D1:1", "Bob", "My favorite sport is tennis."),
            _turn("D1:2", "Alice", "I love hiking in the mountains."),
        )

        memories = Regex().extract(dialogue)

        self.assertEqual(len(memories), 2)
        self.assertIn("possessive", memories[0].meta["matched_patterns"])
        self.assertIn("preference", memories[1].meta["matched_patterns"])


class RegexV2Tests(unittest.TestCase):
    def test_v2_keeps_we_and_contraction_facts_that_v1_missed(self):
        dialogue = _dialogue(
            _turn("D1:1", "Alice", "We made pizza together!"),
            _turn("D1:2", "Bob", "I've worked with Python and C++."),
        )

        self.assertEqual(len(Regex(version="v1").extract(dialogue)), 0)
        self.assertEqual(len(Regex(version="v2").extract(dialogue)), 2)

    def test_v2_state_drops_discourse_fillers_v1_kept(self):
        filler = _dialogue(_turn("D1:1", "Alice", "I'm so glad to hear that!"))
        fact = _dialogue(_turn("D1:2", "Bob", "I'm a middle school teacher."))

        self.assertEqual(len(Regex(version="v1").extract(filler)), 1)
        self.assertEqual(len(Regex(version="v2").extract(filler)), 0)
        self.assertEqual(len(Regex(version="v2").extract(fact)), 1)

    def test_rejects_unknown_version(self):
        with self.assertRaises(ValueError):
            Regex(version="v3")


class TimestampPrefixTests(unittest.TestCase):
    def test_default_output_has_no_prefix(self):
        dialogue = _dialogue(_turn("D1:1", "Alice", "I adopted a cat yesterday."))

        for extractor in (AppendAll(), Regex()):
            memories = extractor.extract(dialogue)
            self.assertEqual(memories[0].text, "Alice: I adopted a cat yesterday.")

    def test_with_timestamp_prepends_iso_date(self):
        dialogue = _dialogue(_turn("D1:1", "Alice", "I adopted a cat yesterday."))

        for extractor in (AppendAll(with_timestamp=True), Regex(with_timestamp=True)):
            memories = extractor.extract(dialogue)
            self.assertEqual(memories[0].text, "[2024-01-01] Alice: I adopted a cat yesterday.")

    def test_falls_back_to_raw_date_when_parse_failed(self):
        turn = _turn("D1:1", "Alice", "I adopted a cat yesterday.")
        turn.timestamp = None
        dialogue = _dialogue(turn)

        memories = AppendAll(with_timestamp=True).extract(dialogue)

        self.assertEqual(memories[0].text, "[1 January 2024] Alice: I adopted a cat yesterday.")


def _spacy_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


@unittest.skipUnless(_spacy_available(), "spaCy en_core_web_sm not installed")
class NERTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.extract.ner import NER

        cls.extractor = NER()

    def test_keeps_entity_turns_and_drops_chitchat(self):
        dialogue = _dialogue(
            _turn("D1:1", "Alice", "Totally agree, so true."),
            _turn("D1:2", "Bob", "I moved to Boston with Marmalade last Tuesday."),
        )

        memories = self.extractor.extract(dialogue)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source_dia_ids, ["D1:2"])
        labels = {label for _, label in memories[0].meta["entities"]}
        self.assertIn("GPE", labels)

    def test_memory_format_matches_append_all(self):
        dialogue = _dialogue(
            _turn("D1:1", "Bob", "I visited Paris in June with my sister Anna."),
        )

        memories = self.extractor.extract(dialogue)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].text, "Bob: I visited Paris in June with my sister Anna.")
        self.assertEqual(memories[0].meta["speaker"], "Bob")
        self.assertEqual(memories[0].meta["session_id"], "session_1")


if __name__ == "__main__":
    unittest.main()
