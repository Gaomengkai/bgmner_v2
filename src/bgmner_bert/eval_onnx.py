from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from .data import (
    TokenizedNerDataset,
    build_bio_label_maps,
    load_label_bases,
    load_ner_jsonl,
)
from .metrics import (
    compute_sequence_metrics,
    labels_from_predictions_and_references,
    safe_classification_report,
)
from .onnx_runtime import build_onnx_session, default_provider_argument


T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_onnx_model(
    onnx_path: Path,
    model_dir: Path,
    dataset_file: Path,
    provider: str = default_provider_argument(),
    batch_size: int = 32,
    max_length: int = 256,
    max_samples: int = 0,
) -> tuple[dict, str]:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    labels_path = dataset_file.parent / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer is required for word_ids alignment.")

    label_bases = load_label_bases(labels_path)
    _, label2id, id2label = build_bio_label_maps(label_bases)
    samples = load_ner_jsonl(dataset_file)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("No dataset samples for evaluation.")

    dataset = TokenizedNerDataset(
        samples=samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    session, provider_chain, available_providers = build_onnx_session(
        onnx_path=onnx_path,
        provider=provider,
    )
    session_providers = list(session.get_providers())
    active_provider = session_providers[0] if session_providers else provider_chain[0]
    input_names = {item.name for item in session.get_inputs()}

    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []
    start = time.perf_counter()

    for batch_features in iter_batches(dataset.features, batch_size):
        collated = collator(batch_features)
        labels = collated.pop("labels").numpy()

        feed = {}
        for name in input_names:
            if name not in collated:
                continue
            feed[name] = collated[name].numpy().astype(np.int64)

        logits = session.run(["logits"], feed)[0]
        batch_true, batch_pred = labels_from_predictions_and_references(
            predictions=logits,
            label_ids=labels,
            id2label=id2label,
        )
        all_true.extend(batch_true)
        all_pred.extend(batch_pred)

    runtime_sec = time.perf_counter() - start
    metrics = compute_sequence_metrics(all_true, all_pred)
    metrics.update(
        {
            "provider": active_provider,
            "requested_provider": provider,
            "provider_chain": provider_chain,
            "session_providers": session_providers,
            "available_providers": available_providers,
            "samples": len(dataset),
            "runtime_sec": runtime_sec,
            "samples_per_sec": len(dataset) / runtime_sec if runtime_sec > 0 else 0.0,
        }
    )
    report = safe_classification_report(all_true, all_pred)
    return metrics, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ONNX NER model on dataset.")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-file", default="data/ner_data/dev.txt")
    parser.add_argument(
        "--provider",
        default=default_provider_argument(),
        help=(
            "onnxruntime execution provider. Supports aliases: "
            "cpu/coreml/cuda/rocm/dml. "
            "You can pass a chain like 'coreml,cpu'; "
            "use 'auto' for platform defaults."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-report", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics, report = evaluate_onnx_model(
        onnx_path=Path(args.onnx_path).resolve(),
        model_dir=Path(args.model_dir).resolve(),
        dataset_file=Path(args.dataset_file).resolve(),
        provider=args.provider,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_report:
        output_report = Path(args.output_report).resolve()
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
