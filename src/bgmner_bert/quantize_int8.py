from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_op_types(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_weight_type(value: str) -> QuantType:
    normalized = value.strip().lower()
    if normalized == "qint8":
        return QuantType.QInt8
    if normalized == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported weight type: {value}")


def infer_output_path(input_onnx: Path, output_onnx: str) -> Path:
    if output_onnx:
        return Path(output_onnx).resolve()
    base = input_onnx.with_suffix("")
    return Path(str(base) + ".int8.dynamic.onnx").resolve()


def quantize_to_int8(
    input_onnx: Path,
    output_onnx: Path,
    weight_type: str = "qint8",
    per_channel: bool = True,
    reduce_range: bool = False,
    op_types: Sequence[str] | None = None,
) -> dict:
    input_onnx = input_onnx.resolve()
    output_onnx = output_onnx.resolve()
    if not input_onnx.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_onnx}")
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        op_types_to_quantize=list(op_types) if op_types else None,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=resolve_weight_type(weight_type),
    )

    result = {
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "weight_type": weight_type,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "op_types": list(op_types) if op_types else [],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to dynamic INT8.")
    parser.add_argument("--input-onnx", required=True, help="FP32 ONNX model path.")
    parser.add_argument(
        "--output-onnx",
        default="",
        help="Output INT8 ONNX path. Default: <input>.int8.dynamic.onnx",
    )
    parser.add_argument(
        "--weight-type",
        default="qint8",
        choices=["qint8", "quint8"],
        help="INT8 weight quantization type.",
    )
    parser.add_argument(
        "--per-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-channel quantization for weights.",
    )
    parser.add_argument(
        "--reduce-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reduced quantization range.",
    )
    parser.add_argument(
        "--op-types",
        default="MatMul,Gemm",
        help="Comma-separated op types to quantize.",
    )
    parser.add_argument(
        "--meta-output",
        default="",
        help="Optional metadata JSON path. Default: sibling quantize_meta.json",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_onnx = Path(args.input_onnx).resolve()
    output_onnx = infer_output_path(input_onnx, args.output_onnx)
    op_types = parse_op_types(args.op_types)

    result = quantize_to_int8(
        input_onnx=input_onnx,
        output_onnx=output_onnx,
        weight_type=args.weight_type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        op_types=op_types,
    )

    if args.meta_output:
        meta_path = Path(args.meta_output).resolve()
    else:
        meta_path = output_onnx.parent / "quantize_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(str(output_onnx))


if __name__ == "__main__":
    main()

