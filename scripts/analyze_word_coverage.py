from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from bgmner_bert.data import load_ner_jsonl
from bgmner_bert.inference_utils import sanitize_word_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tokenization word-id coverage.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-file", default="data/bgm/ner_data/dev.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_dir).resolve(), use_fast=True)
    samples = load_ner_jsonl(Path(args.dataset_file).resolve())

    total_samples = 0
    total_missing_positions = 0
    total_missing_non_o = 0
    missing_label_counter: Counter[str] = Counter()

    for batch in iter_batches(samples, args.batch_size):
        words_batch = [sample.words for sample in batch]
        encoded = tokenizer(
            words_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="np",
        )
        for i, sample in enumerate(batch):
            word_ids = sanitize_word_ids(encoded.word_ids(i))
            present = {wid for wid in word_ids if wid >= 0}
            if not present:
                continue
            max_word_id = max(present)
            missing = [idx for idx in range(max_word_id + 1) if idx not in present]
            total_samples += 1
            total_missing_positions += len(missing)
            for idx in missing:
                if idx >= len(sample.labels):
                    continue
                label = sample.labels[idx]
                missing_label_counter[label] += 1
                if label != "O":
                    total_missing_non_o += 1

    print(f"samples={total_samples}")
    print(f"missing_positions={total_missing_positions}")
    print(f"missing_non_o={total_missing_non_o}")
    ratio = (total_missing_non_o / total_missing_positions) if total_missing_positions else 0.0
    print(f"missing_non_o_ratio={ratio:.6f}")
    print("missing_label_top20=", missing_label_counter.most_common(20))


if __name__ == "__main__":
    main()
