from __future__ import annotations

from typing import Dict, List


def normalize_bio_tags(tags: List[str]) -> List[str]:
    normalized: List[str] = []
    active_type: str | None = None
    for tag in tags:
        if tag == "O":
            normalized.append("O")
            active_type = None
            continue
        if "-" not in tag:
            normalized.append("O")
            active_type = None
            continue
        prefix, label_type = tag.split("-", 1)
        if prefix == "B":
            normalized.append(tag)
            active_type = label_type
            continue
        if prefix == "I":
            if active_type == label_type:
                normalized.append(tag)
            else:
                normalized.append(f"B-{label_type}")
            active_type = label_type
            continue
        normalized.append("O")
        active_type = None
    return normalized


def decode_entities(words: List[str], tags: List[str]) -> Dict[str, List[List[object]]]:
    if len(words) != len(tags):
        raise ValueError(
            f"words/tags length mismatch: {len(words)} vs {len(tags)}"
        )

    normalized = normalize_bio_tags(tags)
    entities: Dict[str, List[List[object]]] = {}

    active_type: str | None = None
    start_idx = -1

    for index, tag in enumerate(normalized + ["O"]):
        if tag == "O":
            if active_type is not None:
                end_idx = index - 1
                text = "".join(words[start_idx : end_idx + 1])
                entities.setdefault(active_type, []).append([text, start_idx, end_idx])
                active_type = None
                start_idx = -1
            continue

        prefix, label_type = tag.split("-", 1)
        if prefix == "B":
            if active_type is not None:
                end_idx = index - 1
                text = "".join(words[start_idx : end_idx + 1])
                entities.setdefault(active_type, []).append([text, start_idx, end_idx])
            active_type = label_type
            start_idx = index
            continue

        if prefix == "I":
            # After normalization, I always belongs to current entity.
            if active_type is None:
                active_type = label_type
                start_idx = index
            continue

    return entities

