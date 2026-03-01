from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer

from .inference_utils import predict_entities_from_token_ids, sanitize_word_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ONNX NER inference.")
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to HF best_model directory (for tokenizer and id2label).",
    )
    parser.add_argument("--text", default="", help="Single input text.")
    parser.add_argument("--input-file", default="", help="Input file, one text per line.")
    parser.add_argument("--output-file", default="", help="Output JSONL path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="onnxruntime execution provider.",
    )
    return parser


def load_inputs(text: str, input_file: str) -> List[str]:
    lines: List[str] = []
    if text:
        lines.append(text)
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    lines.append(line)
    if not lines:
        raise ValueError("Provide --text or --input-file.")
    return lines


def iter_batches(items: List[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def predict_batch(
    session: ort.InferenceSession,
    tokenizer,
    texts: List[str],
    max_length: int,
    id2label: dict[int, str],
) -> List[dict]:
    words_batch = [list(text) for text in texts]
    encoded = tokenizer(
        words_batch,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="np",
    )
    word_ids_batch = [sanitize_word_ids(encoded.word_ids(i)) for i in range(len(texts))]

    feed = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    input_names = {item.name for item in session.get_inputs()}
    feed = {name: value for name, value in feed.items() if name in input_names}

    logits = session.run(["logits"], feed)[0]
    pred_ids = np.argmax(logits, axis=-1).tolist()

    results: List[dict] = []
    for text, words, word_ids, token_pred_ids in zip(
        texts, words_batch, word_ids_batch, pred_ids
    ):
        entities, word_tags, truncated_words = predict_entities_from_token_ids(
            words=words,
            token_label_ids=token_pred_ids,
            word_ids=word_ids,
            id2label=id2label,
        )
        results.append(
            {
                "text": text,
                "truncated_text": "".join(truncated_words),
                "entities": entities,
                "pred_labels": word_tags,
            }
        )
    return results


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    onnx_path = Path(args.onnx_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    available_providers = ort.get_available_providers()
    if args.provider not in available_providers:
        raise RuntimeError(
            f"Provider '{args.provider}' not available. Available: {available_providers}"
        )
    session = ort.InferenceSession(str(onnx_path), providers=[args.provider])

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config = AutoConfig.from_pretrained(model_dir)
    id2label = {int(key): value for key, value in config.id2label.items()}

    inputs = load_inputs(args.text, args.input_file)
    all_results: List[dict] = []
    for batch in iter_batches(inputs, args.batch_size):
        all_results.extend(
            predict_batch(
                session=session,
                tokenizer=tokenizer,
                texts=batch,
                max_length=args.max_length,
                id2label=id2label,
            )
        )

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in all_results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif len(all_results) == 1:
        print(json.dumps(all_results[0], ensure_ascii=False, indent=2))
    else:
        for row in all_results:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

