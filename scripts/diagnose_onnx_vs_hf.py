from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bgmner_bert.data import load_ner_jsonl
from bgmner_bert.inference_utils import sanitize_word_ids, token_ids_to_word_tags
from bgmner_bert.metrics import compute_sequence_metrics


def iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose ONNX vs HF token-classification parity.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--dataset-file", default="data/bgm/ner_data/dev.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    onnx_path = Path(args.onnx_path).resolve()
    dataset_file = Path(args.dataset_file).resolve()

    samples = load_ner_jsonl(dataset_file)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}

    all_true: list[list[str]] = []
    all_hf: list[list[str]] = []
    all_onnx: list[list[str]] = []
    mismatch_sequences = 0
    mismatch_tokens = 0
    total_sequences = 0
    total_tokens = 0

    for batch in iter_batches(samples, args.batch_size):
        words_batch = [sample.words for sample in batch]
        encoded_np = tokenizer(
            words_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="np",
        )
        word_ids_batch = [
            sanitize_word_ids(encoded_np.word_ids(i)) for i in range(len(batch))
        ]
        feed = {
            "input_ids": encoded_np["input_ids"].astype(np.int64),
            "attention_mask": encoded_np["attention_mask"].astype(np.int64),
        }
        feed = {name: value for name, value in feed.items() if name in input_names}

        with torch.no_grad():
            hf_logits = model(
                input_ids=torch.from_numpy(feed["input_ids"]),
                attention_mask=torch.from_numpy(feed["attention_mask"]),
            ).logits.cpu().numpy()
        onnx_logits = session.run(["logits"], feed)[0]

        hf_ids = np.argmax(hf_logits, axis=-1)
        onnx_ids = np.argmax(onnx_logits, axis=-1)

        for sample, word_ids, hf_row, onnx_row in zip(
            batch, word_ids_batch, hf_ids, onnx_ids
        ):
            hf_tags = token_ids_to_word_tags(hf_row.tolist(), word_ids, id2label)
            onnx_tags = token_ids_to_word_tags(onnx_row.tolist(), word_ids, id2label)
            gold_tags = sample.labels[: len(hf_tags)]

            all_true.append(gold_tags)
            all_hf.append(hf_tags)
            all_onnx.append(onnx_tags)

            total_sequences += 1
            if hf_tags != onnx_tags:
                mismatch_sequences += 1

            token_len = min(len(hf_tags), len(onnx_tags))
            total_tokens += token_len
            mismatch_tokens += sum(
                1 for h_tag, o_tag in zip(hf_tags[:token_len], onnx_tags[:token_len]) if h_tag != o_tag
            )

    hf_metrics = compute_sequence_metrics(all_true, all_hf)
    onnx_metrics = compute_sequence_metrics(all_true, all_onnx)

    print(f"samples={total_sequences}")
    print(f"mismatch_sequences={mismatch_sequences}")
    print(
        "mismatch_sequence_ratio="
        f"{(mismatch_sequences / total_sequences if total_sequences else 0.0):.6f}"
    )
    print(f"mismatch_tokens={mismatch_tokens}")
    print(
        "mismatch_token_ratio="
        f"{(mismatch_tokens / total_tokens if total_tokens else 0.0):.6f}"
    )
    print(f"hf_metrics={hf_metrics}")
    print(f"onnx_metrics={onnx_metrics}")


if __name__ == "__main__":
    main()
