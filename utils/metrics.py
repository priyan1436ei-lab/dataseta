"""Classification metrics, calibration calculations, and reporting helpers."""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics across all classes."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

    # Per-class metrics
    per_class = {}
    if class_names is not None:
        target_names = [str(c) for c in class_names]
    else:
        target_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names if len(target_names) == len(np.unique(np.concatenate([y_true, y_pred]))) else None,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "classification_report": report_dict,
    }

    if y_prob is not None:
        ece = compute_expected_calibration_error(y_true, y_prob)
        metrics["expected_calibration_error"] = ece

    return metrics


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
