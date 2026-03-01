from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .bio_decode import decode_entities


def sanitize_word_ids(word_ids: Sequence[int | None]) -> List[int]:
    return [-1 if wid is None else int(wid) for wid in word_ids]


def token_ids_to_word_tags(
    token_label_ids: Sequence[int], word_ids: Sequence[int], id2label: Dict[int, str]
) -> List[str]:
    max_word_id = max((wid for wid in word_ids if wid >= 0), default=-1)
    if max_word_id < 0:
        return []

    word_tags = ["O"] * (max_word_id + 1)
    seen_words = set()
    for token_index, word_id in enumerate(word_ids):
        if word_id < 0 or word_id in seen_words:
            continue
        if token_index >= len(token_label_ids):
            break
        seen_words.add(word_id)
        word_tags[word_id] = id2label.get(int(token_label_ids[token_index]), "O")
    return word_tags


def predict_entities_from_token_ids(
    words: Sequence[str],
    token_label_ids: Sequence[int],
    word_ids: Sequence[int],
    id2label: Dict[int, str],
) -> Tuple[dict, List[str], List[str]]:
    word_tags = token_ids_to_word_tags(token_label_ids, word_ids, id2label)
    truncated_words = list(words[: len(word_tags)])
    entities = decode_entities(truncated_words, word_tags)
    return entities, word_tags, truncated_words
