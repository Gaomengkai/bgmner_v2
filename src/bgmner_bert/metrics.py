from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def labels_from_predictions_and_references(
    predictions: np.ndarray | Sequence,
    label_ids: np.ndarray | Sequence,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    pred_array = np.asarray(predictions)
    label_array = np.asarray(label_ids)

    if pred_array.ndim == 3:
        pred_ids = np.argmax(pred_array, axis=-1)
    elif pred_array.ndim == 2:
        pred_ids = pred_array
    else:
        raise ValueError(
            f"predictions must be rank-2 or rank-3, got rank-{pred_array.ndim}"
        )

    if label_array.ndim != 2:
        raise ValueError(f"label_ids must be rank-2, got rank-{label_array.ndim}")

    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []
    for pred_row, label_row in zip(pred_ids, label_array):
        row_true: List[str] = []
        row_pred: List[str] = []
        for pred_id, gold_id in zip(pred_row, label_row):
            if int(gold_id) == -100:
                continue
            row_true.append(id2label[int(gold_id)])
            row_pred.append(id2label[int(pred_id)])
        all_true.append(row_true)
        all_pred.append(row_pred)
    return all_true, all_pred


def compute_sequence_metrics(
    all_true: Sequence[Sequence[str]], all_pred: Sequence[Sequence[str]]
) -> Dict[str, float]:
    return {
        "precision": float(precision_score(all_true, all_pred)),
        "recall": float(recall_score(all_true, all_pred)),
        "f1": float(f1_score(all_true, all_pred)),
        "accuracy": float(accuracy_score(all_true, all_pred)),
    }


def safe_classification_report(
    all_true: Sequence[Sequence[str]], all_pred: Sequence[Sequence[str]]
) -> str:
    try:
        return classification_report(
            all_true, all_pred, digits=4, zero_division=0
        )
    except TypeError:
        # Compatibility with older seqeval versions.
        return classification_report(all_true, all_pred, digits=4)


def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        predictions, label_ids = eval_prediction
        all_true, all_pred = labels_from_predictions_and_references(
            predictions, label_ids, id2label
        )
        return compute_sequence_metrics(all_true, all_pred)

    return compute_metrics

