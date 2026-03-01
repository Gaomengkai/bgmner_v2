from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from bgmner_bert.data import (
    TokenizedNerDataset,
    build_bio_label_maps,
    load_label_bases,
    load_ner_jsonl,
)
from bgmner_bert.inference_utils import sanitize_word_ids, token_ids_to_word_tags
from bgmner_bert.metrics import (
    compute_sequence_metrics,
    labels_from_predictions_and_references,
)


def iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trainer-style vs word-style evaluation.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", default="data/bgm/ner_data")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def eval_trainer_style(
    model,
    tokenizer,
    dataset_dir: Path,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    label_bases = load_label_bases(dataset_dir / "labels.txt")
    bio_labels, label2id, id2label = build_bio_label_maps(label_bases)
    _ = bio_labels

    dev_samples = load_ner_jsonl(dataset_dir / "dev.txt")
    dev_dataset = TokenizedNerDataset(
        samples=dev_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []
    for feature_batch in iter_batches(dev_dataset.features, batch_size):
        collated = collator(feature_batch)
        labels = collated.pop("labels").numpy()
        inputs = {key: value for key, value in collated.items() if key in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        batch_true, batch_pred = labels_from_predictions_and_references(logits, labels, id2label)
        all_true.extend(batch_true)
        all_pred.extend(batch_pred)
    return compute_sequence_metrics(all_true, all_pred)


def eval_word_style(
    model,
    tokenizer,
    dataset_dir: Path,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    dev_samples = load_ner_jsonl(dataset_dir / "dev.txt")
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []
    for batch in iter_batches(dev_samples, batch_size):
        words_batch = [sample.words for sample in batch]
        encoded = tokenizer(
            words_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        word_ids_batch = [sanitize_word_ids(encoded.word_ids(i)) for i in range(len(batch))]
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1)

        for sample, word_ids, token_pred_ids in zip(batch, word_ids_batch, pred_ids):
            pred_word_tags = token_ids_to_word_tags(token_pred_ids.tolist(), word_ids, id2label)
            gold_word_tags = sample.labels[: len(pred_word_tags)]
            all_true.append(gold_word_tags)
            all_pred.append(pred_word_tags)
    return compute_sequence_metrics(all_true, all_pred)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    trainer_metrics = eval_trainer_style(
        model=model,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    word_metrics = eval_word_style(
        model=model,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"trainer_style={trainer_metrics}")
    print(f"word_style={word_metrics}")


if __name__ == "__main__":
    main()
