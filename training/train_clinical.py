"""Clinical Symptom Risk Modeling Pipeline for PashuRaksha AI.

Trains an XGBoost classifier on preprocessed clinical, physiological, and symptom features.
Handles class imbalance with sample weights.
Saves model to models/clinical_model/model.joblib and outputs metrics to reports/clinical_metrics.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything
from utils.metrics import compute_classification_metrics
from data_processing.prepare_clinical import process_clinical_dataset, ClinicalPreprocessor

logger = get_logger("TrainClinical")


def train_clinical_model(
    data_path: str = "data/clinical/clinical_data.csv",
    output_dir: str = "models/clinical_model",
    reports_dir: str = "reports",
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Preprocesses data, trains XGBoost, evaluates on test set, and saves artifacts."""
    seed_everything(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = process_clinical_dataset(
        file_path=data_path,
        output_dir=output_dir,
        seed=seed,
    )

    classes = [str(c) for c in preprocessor.target_encoder.classes_]
    num_classes = len(classes)
    logger.info(f"Clinical model target classes ({num_classes}): {classes}")

    # Compute sample weights for imbalanced clinical classes
    sample_weights_train = compute_sample_weight("balanced", y_train)

    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
    )

    logger.info("Training XGBoost clinical classifier...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Save model
    model_save_path = Path(output_dir) / "model.joblib"
    joblib.dump(model, model_save_path)
    logger.info(f"Saved clinical XGBoost model to: {model_save_path}")

    # Evaluate on untouched test set
    y_prob_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_prob_test, axis=1)

    metrics = compute_classification_metrics(y_test, y_pred_test, class_names=classes, y_prob=y_prob_test)
    metrics["test_samples_count"] = len(y_test)
    metrics["feature_names"] = preprocessor.feature_names
    metrics["target_classes"] = classes

    # Save metrics JSON
    metrics_path = Path(reports_dir) / "clinical_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved clinical test metrics to: {metrics_path}")

    # Plot Confusion Matrix
    cm_path = Path(reports_dir) / "clinical_confusion_matrix.png"
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        np.array(metrics["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Clinical XGBoost Model: Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Disease", fontweight="semibold")
    plt.ylabel("True Diagnosis", fontweight="semibold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Clinical confusion matrix saved to: {cm_path}")

    logger.info("==================================================")
    logger.info("         CLINICAL MODEL TEST EVALUATION RESULTS   ")
    logger.info("==================================================")
    logger.info(f"Accuracy             : {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Balanced Accuracy     : {metrics['balanced_accuracy'] * 100:.2f}%")
    logger.info(f"Macro Precision       : {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall          : {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1 Score        : {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1 Score     : {metrics['weighted_f1']:.4f}")
    logger.info("==================================================")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Clinical XGBoost Training")
    parser.add_argument("--data", type=str, default="data/clinical/clinical_data.csv", help="Clinical CSV path")
    parser.add_argument("--output-dir", type=str, default="models/clinical_model", help="Model save directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=5, help="Tree max depth")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_clinical_model(
        data_path=args.data,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
