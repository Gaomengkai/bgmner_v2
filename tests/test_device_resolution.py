from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.api import build_parser as build_api_parser  # noqa: E402
from bgmner_bert.benchmark import build_parser as build_benchmark_parser  # noqa: E402
from bgmner_bert.predict import build_parser as build_predict_parser  # noqa: E402
from bgmner_bert.predict import resolve_device  # noqa: E402


class DeviceResolutionTest(unittest.TestCase):
    def test_predict_parser_accepts_mps(self) -> None:
        parser = build_predict_parser()
        args = parser.parse_args(["--model-dir", "runs/x/best_model", "--text", "hello", "--device", "mps"])
        self.assertEqual(args.device, "mps")

    def test_benchmark_parser_accepts_mps(self) -> None:
        parser = build_benchmark_parser()
        args = parser.parse_args(["--model-dir", "runs/x/best_model", "--text", "hello", "--device", "mps"])
        self.assertEqual(args.device, "mps")

    def test_api_parser_accepts_mps(self) -> None:
        parser = build_api_parser()
        args = parser.parse_args(["--model-dir", "runs/x/best_model", "--device", "mps"])
        self.assertEqual(args.device, "mps")

    @patch("bgmner_bert.predict.torch.backends.mps.is_available", return_value=False)
    @patch("bgmner_bert.predict.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda_over_mps(self, _mock_cuda, _mock_mps) -> None:
        self.assertEqual(resolve_device("auto").type, "cuda")

    @patch("bgmner_bert.predict.torch.backends.mps.is_available", return_value=True)
    @patch("bgmner_bert.predict.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_mps(self, _mock_cuda, _mock_mps) -> None:
        self.assertEqual(resolve_device("auto").type, "mps")

    @patch("bgmner_bert.predict.torch.backends.mps.is_available", return_value=False)
    @patch("bgmner_bert.predict.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _mock_cuda, _mock_mps) -> None:
        self.assertEqual(resolve_device("auto").type, "cpu")

    @patch("bgmner_bert.predict.torch.backends.mps.is_available", return_value=False)
    def test_explicit_mps_raises_when_unavailable(self, _mock_mps) -> None:
        with self.assertRaisesRegex(RuntimeError, "MPS requested but not available."):
            resolve_device("mps")

    @patch("bgmner_bert.predict.torch.backends.mps.is_available", return_value=True)
    def test_explicit_mps_when_available(self, _mock_mps) -> None:
        self.assertEqual(resolve_device("mps").type, "mps")


if __name__ == "__main__":
    unittest.main()
