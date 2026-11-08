"""Run with:  python -m unittest discover -s tests -v   (pytest also works)"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder import scoring as S
from promo_finder.llm import parse_json, rerank
from promo_finder.parser import parse_episode, to_seconds
from promo_finder.pipeline import Weights, find_promos

SAMPLE = """EPISODE NUMBER: Episode 2
EPISODE LOGLINE: A bride is dumped at the altar.
CONTINUITY NOTES: Picks up directly from Episode 1.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 00:40.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Wide shot of the nave.

**GERALD (Cold, Cruel):**
Start: (00:01.00) | End: (00:04.00)
"I cannot marry you! You are a liar!"

**JULIANA (Shaken):**
Start: (00:05.00) | End: (00:08.00)
"Who is she?"

**ACTION:**
Start: (00:09.00) | End: (00:20.00)
Juliana stares, stunned, as a stranger bursts through the doors.

**JULIANA (Whispering):**
Start: (00:21.00) | End: (00:30.00)
"You know what I did, don't you? Tell me the truth..."

**SOUND FX:**
Start: (00:30.00) | End: (00:40.00)
Ominous sting into an abrupt cliffhanger cut.

--- SCRIPT END ---"""

UPPER = SAMPLE.replace("SCENE 01\nScene heading", "SCENE 01\nSCENE HEADING").replace("Scene start", "SCENE START")


class ParserTests(unittest.TestCase):
    def test_header_and_counts(self):
        ep = parse_episode(SAMPLE, "episode-002.txt")
        self.assertEqual(ep.number, 2)
        self.assertEqual(len(ep.events), 6)           # first block is NOT swallowed by the scene header
        self.assertEqual(ep.scenes[0].heading, "INT. CATHEDRAL – DAY")
        self.assertTrue(ep.is_continuation)

    def test_times_and_speakers(self):
        ep = parse_episode(SAMPLE)
        d = [e for e in ep.events if e.kind == "dialogue"][0]
        self.assertEqual((d.speaker, d.tone, d.start, d.end), ("GERALD", "Cold, Cruel", 1.0, 4.0))
        self.assertEqual(d.text, "I cannot marry you! You are a liar!")
        self.assertAlmostEqual(to_seconds("01", "04", "50"), 64.5)

    def test_uppercase_scene_labels(self):
        self.assertEqual(len(parse_episode(UPPER).events), 6)


class ScoringTests(unittest.TestCase):
    def test_word_boundaries(self):
        # old code matched substrings: "no" fired on "know", "die" on "studies"
        self.assertIsNone(S.SHOCK.search("She studies the knowledge of kindness"))
        self.assertIsNotNone(S.SHOCK.search("He is a liar and a traitor"))

    def test_duration_fit(self):
        self.assertEqual(S.duration_fit(30), 10.0)
        self.assertLess(S.duration_fit(21), S.duration_fit(30))
        self.assertLess(S.duration_fit(60), S.duration_fit(45))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ep = parse_episode(SAMPLE)

    def test_promos_respect_brief(self):
        promos, stats = find_promos([self.ep], Weights(), per_ep=3, min_comp=0, min_hook=0, min_cliff=0)
        self.assertTrue(promos)
        for p in promos:
            self.assertTrue(20.0 <= p.duration <= 60.0)
            for s in p.segments:
                self.assertTrue(0 <= s.start < s.end <= self.ep.duration)
        for a in promos:                                # no overlapping picks inside an episode
            for b in promos:
                if a is not b:
                    sa, sb = a.segments[0], b.segments[0]
                    self.assertFalse(sa.start < sb.end and sb.start < sa.end)

    def test_ranks_are_ordered(self):
        promos, _ = find_promos([self.ep], per_ep=3, min_comp=0, min_hook=0, min_cliff=0)
        self.assertEqual([p.rank for p in promos], list(range(1, len(promos) + 1)))
        self.assertEqual([p.composite for p in promos], sorted((p.composite for p in promos), reverse=True))


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.ep = parse_episode(SAMPLE)
        self.promos, _ = find_promos([self.ep], per_ep=1, min_comp=0, min_hook=0, min_cliff=0)

    def test_parse_json_tolerates_prose(self):
        self.assertEqual(parse_json('Sure! {"hook": 9} done')["hook"], 9)
        self.assertIsNone(parse_json("no json here"))

    def test_rerank_uses_injected_model_and_survives_garbage(self):
        before = self.promos[0].composite
        good = lambda s, u: json.dumps({"hook": 10, "cliffhanger": 10, "standalone": 10, "why": "great"})
        self.assertEqual(rerank(self.promos, [self.ep], caller=good), 1)
        self.assertEqual(self.promos[0].scorer, "heuristic+llm")
        self.assertGreaterEqual(self.promos[0].composite, before)
        self.assertEqual(rerank(self.promos, [self.ep], caller=lambda s, u: "garbage"), 0)

        def boom(s, u):
            raise TimeoutError
        self.assertEqual(rerank(self.promos, [self.ep], caller=boom), 0)


if __name__ == "__main__":
    unittest.main()
