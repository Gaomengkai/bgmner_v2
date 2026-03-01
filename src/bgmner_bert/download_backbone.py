from __future__ import annotations

import argparse
import re
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def _sanitize_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "backbone"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download backbone model/tokenizer into project directory."
    )
    parser.add_argument(
        "--model-name",
        default="FacebookAI/xlm-roberta-base",
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Final local backbone directory. Default: backbones/<model_name_sanitized>",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_name = args.model_name
    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        save_dir = (Path("backbones") / _sanitize_model_name(model_name)).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "config.json").exists():
        print(f"model_name={model_name}")
        print(f"save_dir={save_dir}")
        print("status=exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
    )
    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))

    print(f"model_name={model_name}")
    print(f"save_dir={save_dir}")


if __name__ == "__main__":
    main()
