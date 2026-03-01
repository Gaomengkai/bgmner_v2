from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.data import (
    align_word_labels_to_tokens,
    build_bio_label_maps,
    truncate_words_and_labels,
)


class DataAlignmentTest(unittest.TestCase):
    def test_build_bio_label_maps(self) -> None:
        labels, label2id, id2label = build_bio_label_maps(["CT", "EP"])
        self.assertEqual(labels, ["O", "B-CT", "I-CT", "B-EP", "I-EP"])
        self.assertEqual(label2id["I-EP"], 4)
        self.assertEqual(id2label[1], "B-CT")

    def test_truncate_words_and_labels(self) -> None:
        words = list("abcdefgh")
        labels = ["O"] * len(words)
        new_words, new_labels = truncate_words_and_labels(words, labels, max_length=6)
        self.assertEqual(len(new_words), 4)
        self.assertEqual(len(new_labels), 4)

    def test_align_word_labels_to_tokens(self) -> None:
        _, label2id, _ = build_bio_label_maps(["CT"])
        word_ids = [None, 0, 1, 1, None, 2, None]
        word_labels = ["B-CT", "I-CT", "O"]
        aligned = align_word_labels_to_tokens(word_ids, word_labels, label2id)
        self.assertEqual(
            aligned,
            [-100, label2id["B-CT"], label2id["I-CT"], -100, -100, label2id["O"], -100],
        )


if __name__ == "__main__":
    unittest.main()
