"""Clinical Feature Explainability Engine for PashuRaksha AI via SHAP.

Computes SHapley Additive exPlanations for the XGBoost clinical risk model.
Generates:
  - Global feature importance summary plot (reports/explanations/shap_summary.png)
  - Local individual case explanation with top contributing physiological/symptom factors.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from data_processing.prepare_clinical import ClinicalPreprocessor

logger = get_logger("ClinicalExplain")


def explain_clinical_model_global(
    model_path: str = "models/clinical_model/model.joblib",
    preprocessor_path: str = "models/clinical_model/preprocessor.joblib",
    clinical_data_path: str = "data/clinical/clinical_data.csv",
    output_dir: str = "reports/explanations",
) -> str:
    """Generates global SHAP feature importance summary plot."""
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or Preprocessor artifact not found.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    preprocessor: ClinicalPreprocessor = joblib.load(preprocessor_path)

    df = pd.read_csv(clinical_data_path)
    X, _ = preprocessor.transform(df)

    feature_names = preprocessor.feature_names
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        # Multiclass: take mean absolute value across classes or plot summary
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            class_names=[str(c) for c in preprocessor.target_encoder.classes_],
            show=False,
            max_display=12,
        )
    else:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            max_display=12,
        )

    plt.title("Clinical Model: Global SHAP Feature Importance", fontsize=13, fontweight="bold", pad=15)
    out_path = Path(output_dir) / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Global SHAP feature summary saved to: {out_path}")
    return str(out_path)


def explain_single_clinical_record(
    record_dict: Dict[str, Any],
    model_path: str = "models/clinical_model/model.joblib",
    preprocessor_path: str = "models/clinical_model/preprocessor.joblib",
) -> List[Dict[str, Any]]:
    """Calculates local feature contributions for an individual animal's clinical profile."""
    model = joblib.load(model_path)
    preprocessor: ClinicalPreprocessor = joblib.load(preprocessor_path)

    X = preprocessor.transform_single_record(record_dict)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_names = preprocessor.feature_names
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    if isinstance(shap_values, list):
        # Specific to predicted class
        class_shap = shap_values[pred_idx][0]
    elif len(shap_values.shape) == 3:
        class_shap = shap_values[0, :, pred_idx]
    else:
        class_shap = shap_values[0]

    # Rank features by absolute contribution
    ranked_indices = np.argsort(np.abs(class_shap))[::-1]
    contributions = []

    for idx in ranked_indices[:6]:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        val = float(class_shap[idx])
        contributions.append({
            "feature": feat_name,
            "impact": round(val, 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        })

    return contributions


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI SHAP Clinical Explainability")
    parser.add_argument("--data", type=str, default="data/clinical/clinical_data.csv", help="Clinical CSV")
    parser.add_argument("--model", type=str, default="models/clinical_model/model.joblib", help="Model path")
    parser.add_argument("--preprocessor", type=str, default="models/clinical_model/preprocessor.joblib", help="Preprocessor path")
    parser.add_argument("--output-dir", type=str, default="reports/explanations", help="Output directory")
    args = parser.parse_args()

    explain_clinical_model_global(args.model, args.preprocessor, args.data, args.output_dir)


if __name__ == "__main__":
    main()
