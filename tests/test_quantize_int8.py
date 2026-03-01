from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from onnxruntime.quantization import QuantType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bgmner_bert.quantize_int8 import (  # noqa: E402
    build_parser,
    infer_output_path,
    parse_op_types,
    quantize_to_int8,
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

    def test_parser_preprocess_flags(self) -> None:
        parser = build_parser()
        args_default = parser.parse_args(["--input-onnx", "in.onnx"])
        self.assertTrue(args_default.preprocess)
        self.assertTrue(args_default.preprocess_skip_symbolic_shape)
        args_disabled = parser.parse_args(["--input-onnx", "in.onnx", "--no-preprocess"])
        self.assertFalse(args_disabled.preprocess)

    def test_quantize_fallback_to_skip_symbolic_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_onnx = tmp_path / "input.onnx"
            output_onnx = tmp_path / "output.onnx"
            input_onnx.write_bytes(b"onnx-placeholder")

            call_args: list[tuple[bool, bool, bool]] = []

            def fake_preprocess(*, input_model, output_model_path, skip_optimization, skip_onnx_shape, skip_symbolic_shape):
                _ = input_model
                call_args.append((skip_optimization, skip_onnx_shape, skip_symbolic_shape))
                if not skip_symbolic_shape:
                    raise Exception("Incomplete symbolic shape inference")
                Path(output_model_path).write_bytes(b"preprocessed")

            with (
                patch("bgmner_bert.quantize_int8.quant_pre_process", side_effect=fake_preprocess),
                patch("bgmner_bert.quantize_int8.quantize_dynamic"),
            ):
                result = quantize_to_int8(
                    input_onnx=input_onnx,
                    output_onnx=output_onnx,
                    preprocess_skip_symbolic_shape=False,
                )

            self.assertEqual(call_args[0], (False, False, False))
            self.assertEqual(call_args[1], (False, False, True))
            self.assertTrue(result["preprocess"])
            self.assertTrue(result["preprocess_skip_symbolic_shape"])


if __name__ == "__main__":
    unittest.main()
