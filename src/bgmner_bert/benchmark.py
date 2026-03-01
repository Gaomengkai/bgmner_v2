from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from .onnx_predict import iter_batches as onnx_iter_batches
from .onnx_predict import predict_batch as onnx_predict_batch
from .onnx_runtime import build_onnx_session, default_provider_argument
from .predict import iter_batches as hf_iter_batches
from .predict import predict_batch as hf_predict_batch
from .predict import resolve_device


def parse_input_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""
    if not line.startswith("{"):
        return line

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line

    if not isinstance(payload, dict) or "text" not in payload:
        return line

    value = payload["text"]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(str(item) for item in value).strip()
    return line


def load_inputs(text: str, input_file: str, max_samples: int = 0) -> List[str]:
    rows: List[str] = []
    if text:
        rows.append(text.strip())

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                value = parse_input_line(raw)
                if value:
                    rows.append(value)
                if max_samples > 0 and len(rows) >= max_samples:
                    break

    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("Provide --text or --input-file.")
    return rows


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize_benchmark_metrics(
    *,
    backend: str,
    texts_per_run: int,
    batch_size: int,
    warmup_runs: int,
    benchmark_runs: int,
    run_total_seconds: Sequence[float],
    batch_latencies_ms: Sequence[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_runtime_sec = float(sum(run_total_seconds))
    total_texts = int(texts_per_run * benchmark_runs)
    total_batches = int(len(batch_latencies_ms))

    metrics: dict[str, Any] = {
        "backend": backend,
        "texts_per_run": texts_per_run,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "total_texts": total_texts,
        "total_batches": total_batches,
        "total_runtime_sec": total_runtime_sec,
        "throughput_texts_per_sec": (
            total_texts / total_runtime_sec if total_runtime_sec > 0 else 0.0
        ),
        "throughput_batches_per_sec": (
            total_batches / total_runtime_sec if total_runtime_sec > 0 else 0.0
        ),
        "avg_text_latency_ms": (
            (total_runtime_sec * 1000.0) / total_texts if total_texts > 0 else 0.0
        ),
        "batch_latency_ms_mean": (
            float(np.mean(batch_latencies_ms)) if batch_latencies_ms else 0.0
        ),
        "batch_latency_ms_p50": _percentile(batch_latencies_ms, 50),
        "batch_latency_ms_p95": _percentile(batch_latencies_ms, 95),
        "batch_latency_ms_p99": _percentile(batch_latencies_ms, 99),
        "run_runtime_sec_mean": (
            float(np.mean(run_total_seconds)) if run_total_seconds else 0.0
        ),
    }
    if extra:
        metrics.update(extra)
    return metrics


def _run_hf_once(
    *,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    model,
    tokenizer,
    device,
    id2label: dict[int, str],
) -> tuple[float, list[float]]:
    batch_latencies_ms: list[float] = []
    started = time.perf_counter()

    for batch in hf_iter_batches(list(texts), batch_size):
        batch_started = time.perf_counter()
        _ = hf_predict_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch,
            max_length=max_length,
            device=device,
            id2label=id2label,
        )
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)

    total_sec = time.perf_counter() - started
    return total_sec, batch_latencies_ms


def _run_onnx_once(
    *,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    session: ort.InferenceSession,
    tokenizer,
    id2label: dict[int, str],
) -> tuple[float, list[float]]:
    batch_latencies_ms: list[float] = []
    started = time.perf_counter()

    for batch in onnx_iter_batches(list(texts), batch_size):
        batch_started = time.perf_counter()
        _ = onnx_predict_batch(
            session=session,
            tokenizer=tokenizer,
            texts=batch,
            max_length=max_length,
            id2label=id2label,
        )
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)

    total_sec = time.perf_counter() - started
    return total_sec, batch_latencies_ms


def benchmark_hf(
    *,
    model_dir: Path,
    texts: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()

    for _ in range(warmup_runs):
        _run_hf_once(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            id2label=id2label,
        )

    run_total_seconds: list[float] = []
    batch_latencies_ms: list[float] = []
    for _ in range(benchmark_runs):
        total_sec, latencies_ms = _run_hf_once(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            id2label=id2label,
        )
        run_total_seconds.append(total_sec)
        batch_latencies_ms.extend(latencies_ms)

    return summarize_benchmark_metrics(
        backend="hf",
        texts_per_run=len(texts),
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        run_total_seconds=run_total_seconds,
        batch_latencies_ms=batch_latencies_ms,
        extra={
            "model_dir": str(model_dir),
            "device": str(resolved_device),
        },
    )


def benchmark_onnx(
    *,
    onnx_path: Path,
    model_dir: Path,
    texts: Sequence[str],
    provider: str,
    batch_size: int,
    max_length: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config = AutoConfig.from_pretrained(model_dir)
    id2label = {int(key): value for key, value in config.id2label.items()}
    session, provider_chain, available_providers = build_onnx_session(
        onnx_path=onnx_path,
        provider=provider,
    )
    session_providers = list(session.get_providers())
    active_provider = session_providers[0] if session_providers else provider_chain[0]

    for _ in range(warmup_runs):
        _run_onnx_once(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            session=session,
            tokenizer=tokenizer,
            id2label=id2label,
        )

    run_total_seconds: list[float] = []
    batch_latencies_ms: list[float] = []
    for _ in range(benchmark_runs):
        total_sec, latencies_ms = _run_onnx_once(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            session=session,
            tokenizer=tokenizer,
            id2label=id2label,
        )
        run_total_seconds.append(total_sec)
        batch_latencies_ms.extend(latencies_ms)

    return summarize_benchmark_metrics(
        backend="onnx",
        texts_per_run=len(texts),
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        run_total_seconds=run_total_seconds,
        batch_latencies_ms=batch_latencies_ms,
        extra={
            "model_dir": str(model_dir),
            "onnx_path": str(onnx_path),
            "provider": active_provider,
            "requested_provider": provider,
            "provider_chain": provider_chain,
            "session_providers": session_providers,
            "available_providers": available_providers,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark NER inference for HF/ONNX.")
    parser.add_argument("--backend", choices=["hf", "onnx"], default="hf")
    parser.add_argument("--model-dir", required=True, help="Path to best_model directory.")
    parser.add_argument("--onnx-path", default="", help="Required when backend=onnx.")
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--text", default="", help="Single input text.")
    parser.add_argument("--input-file", default="", help="Input file, one text/jsonl line per sample.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.backend == "onnx" and not args.onnx_path:
        raise ValueError("--onnx-path is required when --backend onnx.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0.")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be > 0.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    texts = load_inputs(
        text=args.text,
        input_file=args.input_file,
        max_samples=args.max_samples,
    )

    if args.backend == "onnx":
        onnx_path = Path(args.onnx_path).resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        metrics = benchmark_onnx(
            onnx_path=onnx_path,
            model_dir=model_dir,
            texts=texts,
            provider=args.provider,
            batch_size=args.batch_size,
            max_length=args.max_length,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
    else:
        metrics = benchmark_hf(
            model_dir=model_dir,
            texts=texts,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )

    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
