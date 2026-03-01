from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import Dataset


@dataclass(frozen=True)
class NerSample:
    sample_id: str
    words: List[str]
    labels: List[str]


def load_label_bases(labels_path: Path) -> List[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"labels.txt is empty: {labels_path}")
    return labels


def build_bio_label_maps(
    label_bases: Sequence[str],
) -> tuple[List[str], Dict[str, int], Dict[int, str]]:
    bio_labels = ["O"]
    for label in label_bases:
        bio_labels.append(f"B-{label}")
        bio_labels.append(f"I-{label}")
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for idx, label in enumerate(bio_labels)}
    return bio_labels, label2id, id2label


def load_ner_jsonl(path: Path) -> List[NerSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: List[NerSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc

            words = payload.get("text")
            labels = payload.get("labels")
            if not isinstance(words, list) or not isinstance(labels, list):
                raise ValueError(f"{path}:{line_no} requires list fields: text, labels")
            if len(words) != len(labels):
                raise ValueError(
                    f"{path}:{line_no} len(text) != len(labels): {len(words)} != {len(labels)}"
                )
            sample_id = str(payload.get("id", line_no))
            samples.append(
                NerSample(
                    sample_id=sample_id,
                    words=[str(token) for token in words],
                    labels=[str(label) for label in labels],
                )
            )
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def validate_samples(
    samples: Sequence[NerSample], allowed_labels: Sequence[str], dataset_name: str
) -> None:
    allowed = set(allowed_labels)
    for sample in samples:
        for label in sample.labels:
            if label not in allowed:
                raise ValueError(
                    f"{dataset_name} sample_id={sample.sample_id} has unknown label: {label}"
                )


def truncate_words_and_labels(
    words: Sequence[str], labels: Sequence[str], max_length: int
) -> tuple[List[str], List[str]]:
    max_words = max(1, max_length - 2)
    return list(words[:max_words]), list(labels[:max_words])


def align_word_labels_to_tokens(
    word_ids: Sequence[int | None], word_labels: Sequence[str], label2id: Dict[str, int]
) -> List[int]:
    aligned: List[int] = []
    previous_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
            continue
        if word_id != previous_word:
            aligned.append(label2id[word_labels[word_id]])
        else:
            aligned.append(-100)
        previous_word = word_id
    return aligned


class TokenizedNerDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[NerSample],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.features: List[Dict[str, List[int]]] = []
        self.metas: List[Dict[str, object]] = []

        for sample in samples:
            words, labels = truncate_words_and_labels(
                sample.words, sample.labels, max_length=max_length
            )
            encoded = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )

            if not hasattr(encoded, "word_ids"):
                raise RuntimeError("Tokenizer output does not support word_ids().")
            word_ids = encoded.word_ids()
            if word_ids is None:
                raise RuntimeError("word_ids() returned None.")

            aligned_labels = align_word_labels_to_tokens(word_ids, labels, label2id)
            feature = {name: list(values) for name, values in encoded.items()}
            feature["labels"] = aligned_labels
            self.features.append(feature)

            self.metas.append(
                {
                    "id": sample.sample_id,
                    "words": words,
                    "labels": labels,
                    "word_ids": [-1 if wid is None else int(wid) for wid in word_ids],
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]

