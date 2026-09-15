"""Evaluation Engine for PashuRaksha AI Image Classifier.

Evaluates the trained MobileNetV3 model strictly on the untouched test split.
Calculates Accuracy, Balanced Accuracy, Macro/Weighted Precision, Recall, F1, ECE,
per-class metrics, and generates raw/normalized confusion matrix figures.
Outputs:
  - reports/image_test_metrics.json
  - reports/image_classification_report.csv
  - reports/confusion_matrix.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything
from utils.metrics import compute_classification_metrics
from training.train_image import build_mobilenet_v3, get_transforms, LivestockDataset

logger = get_logger("EvaluateImage")


def evaluate_image_model(
    checkpoint_path: str = "models/image_model/best_model.pth",
    split_csv: str = "reports/dataset_split.csv",
    output_metrics_json: str = "reports/image_test_metrics.json",
    output_report_csv: str = "reports/image_classification_report.csv",
    output_cm_png: str = "reports/confusion_matrix.png",
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Loads checkpoint, runs inference on test split, and computes comprehensive test metrics."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Image model checkpoint not found: {checkpoint_path}")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Dataset split CSV not found: {split_csv}")

    # Device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Loading checkpoint from: {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    classes = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]
    temperature = checkpoint.get("temperature", 1.0)
    num_classes = len(classes)

    model = build_mobilenet_v3(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test split
    df_split = pd.read_csv(split_csv)
    test_df = df_split[df_split["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No test samples found in split CSV.")

    logger.info(f"Evaluating on {len(test_df)} untouched test samples...")

    _, eval_transform = get_transforms()
    test_dataset = LivestockDataset(test_df, transform=eval_transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_raw_probs = []
    all_calib_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            raw_prob = F.softmax(logits, dim=1)
            calib_prob = F.softmax(logits / temperature, dim=1)

            preds = torch.argmax(calib_prob, dim=1)

            all_raw_probs.append(raw_prob.cpu().numpy())
            all_calib_probs.append(calib_prob.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    y_prob_raw = np.vstack(all_raw_probs)
    y_prob_calib = np.vstack(all_calib_probs)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names=classes, y_prob=y_prob_calib)
    metrics["raw_ece"] = compute_classification_metrics(y_true, y_pred, class_names=classes, y_prob=y_prob_raw).get("expected_calibration_error", 0.0)
    metrics["calibrated_ece"] = metrics.get("expected_calibration_error", 0.0)
    metrics["learned_temperature"] = temperature
    metrics["test_samples_count"] = len(test_df)

    # Save metrics JSON
    os.makedirs(os.path.dirname(output_metrics_json), exist_ok=True)
    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report CSV
    report_dict = metrics["classification_report"]
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(output_report_csv)

    # Plot Confusion Matrix
    _plot_confusion_matrices(
        np.array(metrics["confusion_matrix"]),
        np.array(metrics["confusion_matrix_normalized"]),
        classes,
        output_cm_png,
    )

    logger.info("==================================================")
    logger.info("          IMAGE MODEL TEST EVALUATION RESULTS     ")
    logger.info("==================================================")
    logger.info(f"Test Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Balanced Accuracy    : {metrics['balanced_accuracy'] * 100:.2f}%")
    logger.info(f"Macro Precision      : {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall         : {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1 Score       : {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1 Score    : {metrics['weighted_f1']:.4f}")
    logger.info(f"Uncalibrated ECE     : {metrics['raw_ece']:.4f}")
    logger.info(f"Calibrated ECE       : {metrics['calibrated_ece']:.4f}")
    logger.info("==================================================")

    return metrics


def _plot_confusion_matrices(cm: np.ndarray, cm_norm: np.ndarray, classes: list, output_path: str) -> None:
    """Generates side-by-side raw count and normalized confusion matrix plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1,
        cbar=False,
    )
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontweight="semibold")
    ax1.set_ylabel("True Label", fontweight="semibold")
    ax1.tick_params(axis="x", rotation=30)

    # Normalized percentages
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax2,
        cbar=True,
    )
    ax2.set_title("Normalized Confusion Matrix (Recall)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Label", fontweight="semibold")
    ax2.set_ylabel("True Label", fontweight="semibold")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Image Model Test Evaluator")
    parser.add_argument("--model", type=str, default="models/image_model/best_model.pth", help="Checkpoint path")
    parser.add_argument("--split-csv", type=str, default="reports/dataset_split.csv", help="Split CSV path")
    parser.add_argument("--output-json", type=str, default="reports/image_test_metrics.json", help="Output JSON path")
    parser.add_argument("--output-csv", type=str, default="reports/image_classification_report.csv", help="Output CSV path")
    parser.add_argument("--output-cm", type=str, default="reports/confusion_matrix.png", help="Output CM PNG path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    evaluate_image_model(
        checkpoint_path=args.model,
        split_csv=args.split_csv,
        output_metrics_json=args.output_json,
        output_report_csv=args.output_csv,
        output_cm_png=args.output_cm,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
