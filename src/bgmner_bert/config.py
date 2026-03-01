from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    config_file: str = ""
    dataset_dir: str = "data/ner_data"
    output_root: str = "runs"
    run_name: str = ""
    model_name: str = "FacebookAI/xlm-roberta-base"
    backbones_dir: str = "backbones"
    max_length: int = 256
    seed: int = 42
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    early_stopping_patience: int = 2
    report_to: str = "none"


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _allowed_config_keys() -> set[str]:
    return {item.name for item in fields(TrainConfig)}


def _load_config_overrides(config_file: str) -> dict[str, Any]:
    if not config_file:
        return {}

    path = Path(config_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")

    allowed = _allowed_config_keys()
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return payload


def build_train_parser(defaults: TrainConfig | None = None) -> argparse.ArgumentParser:
    cfg = defaults or TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train NER with BERT-like backbones + AutoModelForTokenClassification."
    )
    parser.add_argument(
        "--config-file",
        default=cfg.config_file,
        help="Path to external JSON config. CLI args override JSON values.",
    )
    parser.add_argument("--dataset-dir", default=cfg.dataset_dir)
    parser.add_argument("--output-root", default=cfg.output_root)
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument(
        "--backbones-dir",
        default=cfg.backbones_dir,
        help="Directory for local backbone models. HF ids are downloaded here automatically.",
    )
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num-train-epochs", type=float, default=cfg.num_train_epochs)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=cfg.per_device_train_batch_size,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=cfg.per_device_eval_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=cfg.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--logging-steps", type=int, default=cfg.logging_steps)
    parser.add_argument("--save-total-limit", type=int, default=cfg.save_total_limit)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=cfg.dataloader_num_workers,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=cfg.early_stopping_patience,
    )
    parser.add_argument(
        "--report-to",
        default=cfg.report_to,
        help="Set to 'none' to disable external reporting.",
    )
    return parser


def parse_train_args(argv: list[str] | None = None) -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-file", default="")
    pre_args, _ = pre_parser.parse_known_args(argv)

    merged_values = asdict(TrainConfig())
    overrides = _load_config_overrides(pre_args.config_file)
    merged_values.update(overrides)
    if pre_args.config_file:
        merged_values["config_file"] = str(Path(pre_args.config_file).resolve())

    parser = build_train_parser(TrainConfig(**merged_values))
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        config_file=str(Path(args.config_file).resolve()) if args.config_file else "",
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        run_name=args.run_name or default_run_name(),
        model_name=args.model_name,
        backbones_dir=args.backbones_dir,
        max_length=args.max_length,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        early_stopping_patience=args.early_stopping_patience,
        report_to=args.report_to,
    )
    return cfg


def train_config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
