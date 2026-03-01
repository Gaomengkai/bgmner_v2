from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.export_onnx import (  # noqa: E402
    infer_optimized_output_path,
    infer_output_path,
    resolve_graph_optimization_level,
)


class ExportOnnxHelpersTest(unittest.TestCase):
    def test_infer_output_path_default_best_model(self) -> None:
        model_dir = Path("D:/repo/runs/exp1/best_model")
        output = infer_output_path(model_dir, "")
        self.assertTrue(str(output).replace("\\", "/").endswith("runs/exp1/onnx/model.onnx"))

    def test_infer_optimized_output_path_default(self) -> None:
        output = infer_optimized_output_path(Path("D:/repo/model.onnx"), "")
        self.assertTrue(str(output).endswith("model.opt.onnx"))

    def test_resolve_graph_optimization_level_invalid(self) -> None:
        with self.assertRaises(ValueError):
            resolve_graph_optimization_level("invalid")


if __name__ == "__main__":
    unittest.main()
