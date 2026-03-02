from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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


def load_texts(texts: list[str], input_file: str, max_samples: int) -> list[str]:
    rows: list[str] = []

    for text in texts:
        value = text.strip()
        if value:
            rows.append(value)

    if input_file:
        path = Path(input_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                value = parse_input_line(raw_line)
                if value:
                    rows.append(value)
                if max_samples > 0 and len(rows) >= max_samples:
                    break

    if max_samples > 0:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError("Provide --text or --input-file with at least one sample.")
    return rows


def iter_batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in raw_headers:
        if ":" not in raw:
            raise ValueError(f"Invalid --header value: {raw!r}. Expected format: Key:Value")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --header value: {raw!r}. Header key is empty.")
        headers[key] = value
    return headers


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    headers: dict[str, str],
) -> tuple[int, str, float, float]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=request_headers,
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.getcode())
            process_ms_header = response.headers.get("X-Process-Time-Ms")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
        process_ms_header = exc.headers.get("X-Process-Time-Ms")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    process_ms = 0.0
    if process_ms_header:
        try:
            process_ms = float(process_ms_header)
        except ValueError:
            process_ms = 0.0
    return status, body, elapsed_ms, process_ms


def run_once(
    *,
    url: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    payload_mode: str,
    timeout_sec: float,
    headers: dict[str, str],
    continue_on_error: bool,
) -> dict[str, Any]:
    request_latencies_ms: list[float] = []
    process_times_ms: list[float] = []
    failed_requests = 0
    succeeded_texts = 0
    error_samples: list[str] = []
    request_count = 0

    started = time.perf_counter()
    for batch_index, batch in enumerate(iter_batches(texts, batch_size), start=1):
        request_count += 1
        if payload_mode == "items":
            payload: dict[str, Any] = {
                "items": [{"id": i, "text": text} for i, text in enumerate(batch)],
                "batch_size": batch_size,
                "max_length": max_length,
            }
        else:
            payload = {
                "texts": batch,
                "batch_size": batch_size,
                "max_length": max_length,
            }

        status, body, latency_ms, process_ms = post_json(
            url=url,
            payload=payload,
            timeout_sec=timeout_sec,
            headers=headers,
        )
        request_latencies_ms.append(latency_ms)
        if process_ms > 0:
            process_times_ms.append(process_ms)

        if status != 200:
            failed_requests += 1
            if len(error_samples) < 5:
                error_samples.append(f"batch={batch_index} status={status} body={body[:500]}")
            if not continue_on_error:
                raise RuntimeError(
                    f"Request failed at batch={batch_index}, status={status}, body={body[:500]}"
                )
            continue

        try:
            payload_obj = json.loads(body)
        except json.JSONDecodeError:
            failed_requests += 1
            if len(error_samples) < 5:
                error_samples.append(
                    f"batch={batch_index} status=200 invalid_json body={body[:500]}"
                )
            if not continue_on_error:
                raise RuntimeError(f"Invalid JSON response at batch={batch_index}: {body[:500]}")
            continue

        if isinstance(payload_obj, dict):
            if isinstance(payload_obj.get("count"), int):
                succeeded_texts += int(payload_obj["count"])
            elif isinstance(payload_obj.get("results"), list):
                succeeded_texts += len(payload_obj["results"])
            else:
                succeeded_texts += len(batch)
        else:
            succeeded_texts += len(batch)

    total_sec = time.perf_counter() - started
    return {
        "total_sec": total_sec,
        "request_latencies_ms": request_latencies_ms,
        "process_times_ms": process_times_ms,
        "failed_requests": failed_requests,
        "succeeded_texts": succeeded_texts,
        "request_count": request_count,
        "error_samples": error_samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch extraction API (/predict compatible).",
        epilog=(
            "Run with mamba env example:\n"
            "  mamba run -n bgmner python scripts/benchmark_api_batch.py "
            "--url http://127.0.0.1:8000/predict --input-file data/ner_data/dev.txt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="Full predict endpoint URL.")
    parser.add_argument("--text", action="append", default=[], help="Input text (repeatable).")
    parser.add_argument(
        "--input-file",
        default="",
        help="Input file, one sample per line. Supports plain text or JSONL with text field.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--payload-mode", choices=["texts", "items"], default="texts")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra request header, format: Key:Value (repeatable).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue benchmark even if some requests fail.",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be > 0")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be > 0")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)

    headers = parse_headers(args.header)
    texts = load_texts(args.text, args.input_file, args.max_samples)

    for _ in range(args.warmup_runs):
        _ = run_once(
            url=args.url,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            payload_mode=args.payload_mode,
            timeout_sec=args.timeout_sec,
            headers=headers,
            continue_on_error=args.continue_on_error,
        )

    run_times_sec: list[float] = []
    request_latencies_ms: list[float] = []
    process_times_ms: list[float] = []
    failed_requests = 0
    succeeded_texts = 0
    total_requests = 0
    error_samples: list[str] = []

    for _ in range(args.benchmark_runs):
        result = run_once(
            url=args.url,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            payload_mode=args.payload_mode,
            timeout_sec=args.timeout_sec,
            headers=headers,
            continue_on_error=args.continue_on_error,
        )
        run_times_sec.append(float(result["total_sec"]))
        request_latencies_ms.extend(result["request_latencies_ms"])
        process_times_ms.extend(result["process_times_ms"])
        failed_requests += int(result["failed_requests"])
        succeeded_texts += int(result["succeeded_texts"])
        total_requests += int(result["request_count"])
        for item in result["error_samples"]:
            if len(error_samples) < 5:
                error_samples.append(str(item))

    total_runtime_sec = sum(run_times_sec)
    requested_texts = len(texts) * args.benchmark_runs
    successful_requests = total_requests - failed_requests

    metrics: dict[str, Any] = {
        "url": args.url,
        "payload_mode": args.payload_mode,
        "dataset_texts_per_run": len(texts),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "total_requested_texts": requested_texts,
        "total_succeeded_texts": succeeded_texts,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "request_success_rate": (
            successful_requests / total_requests if total_requests > 0 else 0.0
        ),
        "text_success_rate": (
            succeeded_texts / requested_texts if requested_texts > 0 else 0.0
        ),
        "total_runtime_sec": total_runtime_sec,
        "throughput_texts_per_sec": (
            succeeded_texts / total_runtime_sec if total_runtime_sec > 0 else 0.0
        ),
        "throughput_requests_per_sec": (
            total_requests / total_runtime_sec if total_runtime_sec > 0 else 0.0
        ),
        "request_latency_ms_mean": (
            sum(request_latencies_ms) / len(request_latencies_ms)
            if request_latencies_ms
            else 0.0
        ),
        "request_latency_ms_p50": percentile(request_latencies_ms, 50),
        "request_latency_ms_p95": percentile(request_latencies_ms, 95),
        "request_latency_ms_p99": percentile(request_latencies_ms, 99),
        "api_process_time_ms_mean": (
            sum(process_times_ms) / len(process_times_ms) if process_times_ms else 0.0
        ),
        "api_process_time_ms_p50": percentile(process_times_ms, 50),
        "api_process_time_ms_p95": percentile(process_times_ms, 95),
        "api_process_time_ms_p99": percentile(process_times_ms, 99),
        "run_time_sec_mean": (sum(run_times_sec) / len(run_times_sec) if run_times_sec else 0.0),
        "error_samples": error_samples,
    }

    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
