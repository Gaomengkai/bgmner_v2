from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser


def infer_output_path(model_dir: Path, output_path: str) -> Path:
    if output_path:
        return Path(output_path).resolve()
    if model_dir.name == "best_model":
        return (model_dir.parent / "onnx" / "model.onnx").resolve()
    return (model_dir / "onnx" / "model.onnx").resolve()


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

    metadata = {
        "model_dir": str(model_dir),
        "onnx_path": str(output_path),
        "opset": args.opset,
    }
    (output_path.parent / "export_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
