from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .inference_utils import predict_entities_from_token_ids, sanitize_word_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HF NER inference.")
    parser.add_argument("--model-dir", required=True, help="Path to best_model directory.")
    parser.add_argument("--text", default="", help="Single input text.")
    parser.add_argument("--input-file", default="", help="Input file, one text per line.")
    parser.add_argument("--output-file", default="", help="Output JSONL path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    id2label: dict[int, str],
) -> List[dict]:
    words_batch = [list(text) for text in texts]
    encoded = tokenizer(
        words_batch,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    word_ids_batch = [sanitize_word_ids(encoded.word_ids(i)) for i in range(len(texts))]
    model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.no_grad():
        logits = model(**model_inputs).logits
    pred_ids = logits.argmax(dim=-1).cpu().numpy().tolist()

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
    inputs = load_inputs(args.text, args.input_file)

    model_dir = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    all_results: List[dict] = []
    for batch in iter_batches(inputs, args.batch_size):
        all_results.extend(
            predict_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch,
                max_length=args.max_length,
                device=device,
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
