from __future__ import annotations

import sys
import unittest
from pathlib import Path

from onnxruntime.quantization import QuantType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.quantize_int8 import (  # noqa: E402
    infer_output_path,
    parse_op_types,
    resolve_weight_type,
)


class QuantizeInt8HelpersTest(unittest.TestCase):
    def test_parse_op_types(self) -> None:
        self.assertEqual(parse_op_types("MatMul, Gemm , ,Attention"), ["MatMul", "Gemm", "Attention"])

    def test_resolve_weight_type(self) -> None:
        self.assertEqual(resolve_weight_type("qint8"), QuantType.QInt8)
        self.assertEqual(resolve_weight_type("QUINT8"), QuantType.QUInt8)
        with self.assertRaises(ValueError):
            resolve_weight_type("int4")

    def test_infer_output_path(self) -> None:
        input_path = Path("D:/tmp/model.fp32.onnx")
        inferred = infer_output_path(input_path, "")
        self.assertTrue(str(inferred).endswith("model.fp32.int8.dynamic.onnx"))
        explicit = infer_output_path(input_path, "D:/tmp/custom.onnx")
        self.assertTrue(str(explicit).endswith("custom.onnx"))


if __name__ == "__main__":
    unittest.main()
