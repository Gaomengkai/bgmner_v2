from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import parse_train_args, train_config_to_dict
from .data import (
    TokenizedNerDataset,
    build_bio_label_maps,
    load_label_bases,
    load_ner_jsonl,
    validate_samples,
)
from .inference_utils import predict_entities_from_token_ids
from .metrics import (
    build_compute_metrics,
    labels_from_predictions_and_references,
    safe_classification_report,
)
from .download_backbone import main as download_backbone_main

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _detect_device() -> None:
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        LOGGER.info("CUDA available: %s", torch.cuda.get_device_name(device_index))
        LOGGER.info("cuda device count: %s", torch.cuda.device_count())
    else:
        LOGGER.info("CUDA unavailable, training on CPU.")


def _sanitize_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "backbone"


def _resolve_model_source(model_name: str, backbones_dir: Path) -> Path:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path.resolve()

    target_dir = (backbones_dir / _sanitize_model_name(model_name)).resolve()
    if (target_dir / "config.json").exists():
        LOGGER.info("Use existing local backbone: %s", target_dir)
        return target_dir

    LOGGER.info("Download backbone '%s' -> %s", model_name, target_dir)
    download_backbone_main(
        [
            "--model-name",
            model_name,
            "--save-dir",
            str(target_dir),
        ]
    )
    return target_dir


def _save_dev_predictions(
    path: Path,
    dataset: TokenizedNerDataset,
    prediction_ids: np.ndarray,
    id2label: dict[int, str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for meta, token_pred_ids in zip(dataset.metas, prediction_ids):
            entities, word_tags, truncated_words = predict_entities_from_token_ids(
                words=meta["words"],
                token_label_ids=token_pred_ids.tolist(),
                word_ids=meta["word_ids"],
                id2label=id2label,
            )
            record = {
                "id": meta["id"],
                "text": "".join(truncated_words),
                "entities": entities,
                "pred_labels": word_tags,
                "gold_labels": meta["labels"][: len(truncated_words)],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = parse_train_args(argv)
    dataset_dir = Path(cfg.dataset_dir).resolve()
    backbones_dir = Path(cfg.backbones_dir).resolve()
    model_source = _resolve_model_source(cfg.model_name, backbones_dir)
    run_dir = (Path(cfg.output_root) / cfg.run_name).resolve()
    trainer_output_dir = run_dir / "trainer_output"
    best_model_dir = run_dir / "best_model"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    onnx_dir = run_dir / "onnx"
    meta_dir = run_dir / "meta"
    _ensure_dirs(
        backbones_dir,
        trainer_output_dir,
        best_model_dir,
        metrics_dir,
        predictions_dir,
        onnx_dir,
        meta_dir,
    )

    set_seed(cfg.seed)
    _detect_device()

    labels_path = dataset_dir / "labels.txt"
    train_path = dataset_dir / "train.txt"
    dev_path = dataset_dir / "dev.txt"

    label_bases = load_label_bases(labels_path)
    bio_labels, label2id, id2label = build_bio_label_maps(label_bases)

    train_samples = load_ner_jsonl(train_path)
    dev_samples = load_ner_jsonl(dev_path)
    validate_samples(train_samples, bio_labels, dataset_name="train")
    validate_samples(dev_samples, bio_labels, dataset_name="dev")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer is required for word_ids alignment.")

    train_dataset = TokenizedNerDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=cfg.max_length,
    )
    dev_dataset = TokenizedNerDataset(
        samples=dev_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=cfg.max_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(bio_labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    report_to = [] if cfg.report_to.lower() == "none" else [cfg.report_to]

    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        remove_unused_columns=False,
        report_to=report_to,
    )

    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=callbacks,
    )

    LOGGER.info("Training start. run_dir=%s", run_dir)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    prediction_output = trainer.predict(dev_dataset)
    pred_ids = np.argmax(prediction_output.predictions, axis=-1)
    all_true, all_pred = labels_from_predictions_and_references(
        prediction_output.predictions, prediction_output.label_ids, id2label
    )
    report_text = safe_classification_report(all_true, all_pred)
    (metrics_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    _save_dev_predictions(
        path=predictions_dir / "dev_predictions.jsonl",
        dataset=dev_dataset,
        prediction_ids=pred_ids,
        id2label=id2label,
    )

    _write_json(
        metrics_dir / "eval_metrics.json",
        {key: float(value) if isinstance(value, (int, float)) else value for key, value in eval_metrics.items()},
    )
    _write_json(
        metrics_dir / "train_metrics.json",
        {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in train_result.metrics.items()
        },
    )
    _write_json(meta_dir / "train_args.json", train_config_to_dict(cfg))
    _write_json(
        meta_dir / "label_mappings.json",
        {
            "label_bases": label_bases,
            "bio_labels": bio_labels,
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        },
    )
    trainer.state.save_to_json(str(meta_dir / "trainer_state.json"))

    LOGGER.info("Training completed. Best model: %s", best_model_dir)


if __name__ == "__main__":
    main()
