from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.benchmark import (  # noqa: E402
    _validate_args,
    build_parser,
    load_inputs,
    parse_input_line,
    summarize_benchmark_metrics,
)


class BenchmarkHelpersTest(unittest.TestCase):
    def test_parse_input_line_json_variants(self) -> None:
        self.assertEqual(parse_input_line("plain text"), "plain text")
        self.assertEqual(parse_input_line('{"text":"abc"}'), "abc")
        self.assertEqual(parse_input_line('{"text":["a","b","c"]}'), "abc")

    def test_load_inputs_supports_jsonl_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.txt"
            rows = [
                '{"id":1,"text":["你","好"]}',
                '{"id":2,"text":"world"}',
                "plain line",
            ]
            path.write_text("\n".join(rows), encoding="utf-8")
            loaded = load_inputs(text="", input_file=str(path), max_samples=2)
            self.assertEqual(loaded, ["你好", "world"])

    def test_summarize_benchmark_metrics(self) -> None:
        metrics = summarize_benchmark_metrics(
            backend="onnx",
            texts_per_run=40,
            batch_size=8,
            warmup_runs=1,
            benchmark_runs=2,
            run_total_seconds=[1.0, 1.0],
            batch_latencies_ms=[100.0, 150.0, 200.0, 250.0],
            extra={"provider": "CPUExecutionProvider"},
        )
        self.assertEqual(metrics["backend"], "onnx")
        self.assertEqual(metrics["total_texts"], 80)
        self.assertAlmostEqual(metrics["throughput_texts_per_sec"], 40.0, places=6)
        self.assertAlmostEqual(metrics["batch_latency_ms_p50"], 175.0, places=6)
        self.assertEqual(metrics["provider"], "CPUExecutionProvider")

    def test_validate_args_requires_onnx_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--backend",
                "onnx",
                "--model-dir",
                "runs/x/best_model",
                "--text",
                "hello",
            ]
        )
        with self.assertRaises(ValueError):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
