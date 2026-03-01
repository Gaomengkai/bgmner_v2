from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class TokenClassifierOnnxWrapper(torch.nn.Module):
    def __init__(self, model: AutoModelForTokenClassification) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export HF token classifier to ONNX.")
    parser.add_argument("--model-dir", required=True, help="Path to best_model directory.")
    parser.add_argument(
        "--output-path",
        default="",
        help="Output ONNX file path. Default: ../onnx/model.onnx next to best_model.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize exported ONNX graph via onnxruntime.",
    )
    parser.add_argument(
        "--optimize-level",
        default="all",
        choices=["basic", "extended", "all"],
        help="Optimization level used when --optimize is enabled.",
    )
    parser.add_argument(
        "--optimize-output-path",
        default="",
        help="Optimized ONNX output path. Default: <output>.opt.onnx",
    )
    return parser


def infer_output_path(model_dir: Path, output_path: str) -> Path:
    if output_path:
        return Path(output_path).resolve()
    if model_dir.name == "best_model":
        return (model_dir.parent / "onnx" / "model.onnx").resolve()
    return (model_dir / "onnx" / "model.onnx").resolve()


def infer_optimized_output_path(output_path: Path, optimize_output_path: str) -> Path:
    if optimize_output_path:
        return Path(optimize_output_path).resolve()
    base = output_path.with_suffix("")
    return Path(str(base) + ".opt.onnx").resolve()


def resolve_graph_optimization_level(level: str) -> ort.GraphOptimizationLevel:
    normalized = level.strip().lower()
    if normalized == "basic":
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if normalized == "extended":
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if normalized == "all":
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    raise ValueError(f"Unsupported optimize level: {level}")


def optimize_onnx_graph(
    input_path: Path,
    output_path: Path,
    optimize_level: str = "all",
) -> Path:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Optimized output path must be different from input ONNX path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = resolve_graph_optimization_level(optimize_level)
    options.optimized_model_filepath = str(output_path)

    available = ort.get_available_providers()
    provider = "CPUExecutionProvider" if "CPUExecutionProvider" in available else None
    if provider:
        ort.InferenceSession(str(input_path), sess_options=options, providers=[provider])
    else:
        ort.InferenceSession(str(input_path), sess_options=options)
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    output_path = infer_output_path(model_dir, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    dummy_words = [list("测试ONNX导出")]
    tokenized = tokenizer(
        dummy_words,
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
        padding=True,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    wrapper = TokenClassifierOnnxWrapper(model)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    optimized_onnx_path = ""
    if args.optimize:
        optimized_output_path = infer_optimized_output_path(
            output_path=output_path,
            optimize_output_path=args.optimize_output_path,
        )
        optimized_onnx_path = str(
            optimize_onnx_graph(
                input_path=output_path,
                output_path=optimized_output_path,
                optimize_level=args.optimize_level,
            )
        )

    metadata = {
        "model_dir": str(model_dir),
        "onnx_path": str(output_path),
        "opset": args.opset,
        "optimize": args.optimize,
        "optimize_level": args.optimize_level,
        "optimized_onnx_path": optimized_onnx_path,
    }
    (output_path.parent / "export_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(optimized_onnx_path or str(output_path))


if __name__ == "__main__":
    main()
