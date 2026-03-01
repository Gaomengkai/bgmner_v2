from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
    sample_id = row.get("id", 0)
    try:
        order = int(sample_id)
    except (TypeError, ValueError):
        order = 0
    return order, str(sample_id)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _copy_labels(src: Path, dst: Path) -> None:
    labels = [line.strip() for line in src.read_text(encoding="utf-8").splitlines()]
    labels = [line for line in labels if line]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy dataset into local project and sort train/dev by id."
    )
    parser.add_argument("--src-dir", default="../data/bgm/ner_data")
    parser.add_argument("--dst-dir", default="data/ner_data")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset dir not found: {src_dir}")

    for split in ("train", "dev"):
        src_file = src_dir / f"{split}.txt"
        if not src_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {src_file}")
        rows = _load_jsonl(src_file)
        rows_sorted = sorted(rows, key=_sort_key)
        _write_jsonl(dst_dir / f"{split}.txt", rows_sorted)

    labels_src = src_dir / "labels.txt"
    if not labels_src.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_src}")
    _copy_labels(labels_src, dst_dir / "labels.txt")

    print(f"src_dir={src_dir}")
    print(f"dst_dir={dst_dir}")
    print("status=ok")


if __name__ == "__main__":
    main()
