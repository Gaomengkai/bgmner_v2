from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.bio_decode import decode_entities, normalize_bio_tags


class BioDecodeTest(unittest.TestCase):
    def test_normalize_repairs_invalid_i_prefix(self) -> None:
        tags = ["I-CT", "I-CT", "O", "I-EP"]
        normalized = normalize_bio_tags(tags)
        self.assertEqual(normalized, ["B-CT", "I-CT", "O", "B-EP"])

    def test_decode_entities(self) -> None:
        words = list("abcde")
        tags = ["B-CT", "I-CT", "O", "B-EP", "I-EP"]
        entities = decode_entities(words, tags)
        self.assertEqual(entities["CT"][0], ["ab", 0, 1])
        self.assertEqual(entities["EP"][0], ["de", 3, 4])


if __name__ == "__main__":
    unittest.main()
