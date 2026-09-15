"""Offline Clinical Prediction Engine for PashuRaksha AI.

Performs disease risk scoring and classification on individual animal health records.
Outputs predicted disease probabilities, clinical risk score (0-100), and decision notes.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from data_processing.prepare_clinical import ClinicalPreprocessor

logger = get_logger("ClinicalPredict")


def predict_clinical(
    record_dict: Dict[str, Any],
    model_path: str = "models/clinical_model/model.joblib",
    preprocessor_path: str = "models/clinical_model/preprocessor.joblib",
) -> Dict[str, Any]:
    """Runs clinical model inference on a single animal profile."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Clinical model not found: {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor: ClinicalPreprocessor = joblib.load(preprocessor_path)

    X = preprocessor.transform_single_record(record_dict)
    probs = model.predict_proba(X)[0]
    classes = [str(c) for c in preprocessor.target_encoder.classes_]

    sorted_indices = np.argsort(probs)[::-1]
    top_pred_idx = sorted_indices[0]
    predicted_condition = classes[top_pred_idx]
    top_prob = float(probs[top_pred_idx])

    disease_probs = {classes[i]: round(float(probs[i]), 4) for i in sorted_indices}

    # Clinical risk score calculation (0-100)
    # If predicted healthy, risk is low. Otherwise risk scales with probability and temperature deviation
    is_healthy = predicted_condition.lower() in ["healthy_cattle", "healthy", "normal"]
    
    temp = float(record_dict.get("body_temperature_c", record_dict.get("temperature", 38.6)))
    temp_elevation = max(0.0, temp - 39.0)  # fever above 39.0 C
    
    if is_healthy:
        clinical_risk_score = min(25, int(top_prob * 15 + temp_elevation * 10))
    else:
        # Base risk from probability + physiological elevation
        base_risk = top_prob * 70
        fever_penalty = min(25, temp_elevation * 15)
        clinical_risk_score = min(100, int(base_risk + fever_penalty))

    result = {
        "predicted_condition": predicted_condition,
        "clinical_probability": round(top_prob, 4),
        "disease_probabilities": disease_probs,
        "clinical_risk_score": clinical_risk_score,
        "screening_note": "AI clinical screening result. Requires veterinary clinical correlation.",
        "disclaimer": "This is an AI screening prototype and not a replacement for veterinary diagnosis.",
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Clinical Prediction CLI")
    parser.add_argument("--age", type=float, default=3.5, help="Age in years")
    parser.add_argument("--breed", type=str, default="Gir", help="Breed")
    parser.add_argument("--sex", type=str, default="Female", help="Sex")
    parser.add_argument("--weight", type=float, default=420.0, help="Weight in kg")
    parser.add_argument("--temperature", type=float, default=40.5, help="Body temperature in Celsius")
    parser.add_argument("--heart-rate", type=int, default=85, help="Heart rate bpm")
    parser.add_argument("--respiratory-rate", type=int, default=34, help="Respiratory rate bpm")
    parser.add_argument("--milk-drop", type=float, default=40.0, help="Milk yield drop percentage")
    parser.add_argument("--appetite", type=str, default="Reduced", help="Appetite (Normal/Reduced/Anorexia)")
    parser.add_argument("--activity", type=str, default="Lethargic", help="Activity (Active/Normal/Lethargic/Recumbent)")
    parser.add_argument("--vaccination", type=str, default="Overdue", help="Vaccination status")
    parser.add_argument("--symptoms", type=str, default="fever,skin_nodules,loss_of_appetite", help="Comma-separated symptoms")
    parser.add_argument("--model", type=str, default="models/clinical_model/model.joblib", help="Model path")
    parser.add_argument("--preprocessor", type=str, default="models/clinical_model/preprocessor.joblib", help="Preprocessor path")
    args = parser.parse_args()

    record = {
        "age_years": args.age,
        "breed": args.breed,
        "sex": args.sex,
        "weight_kg": args.weight,
        "body_temperature_c": args.temperature,
        "heart_rate_bpm": args.heart_rate,
        "respiratory_rate_bpm": args.respiratory_rate,
        "milk_yield_drop_pct": args.milk_drop,
        "appetite": args.appetite,
        "activity_level": args.activity,
        "vaccination_status": args.vaccination,
        "symptoms": args.symptoms,
    }

    res = predict_clinical(record, args.model, args.preprocessor)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
