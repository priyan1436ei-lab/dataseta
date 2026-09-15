"""Multimodal Decision-Support & Fusion Engine for PashuRaksha AI.

Combines:
  1. Calibrated Image Disease Probabilities & Latent Embeddings
  2. Clinical Symptom & Physiological XGBoost Predictions
  3. Anomaly / Out-of-Distribution Status
  4. Model Agreement & Discrepancy Diagnostics
  5. 0–100 Composite Screening Risk Score & Categorical Risk Level
  6. Clinical Safety Overrides for High-Acuity Symptoms

Outputs structured decision-support JSON with top-K candidate diseases,
contributing factors, and explicit veterinary referral recommendations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from inference.image_predict import predict_image
from inference.clinical_predict import predict_clinical
from explainability.clinical_explain import explain_single_clinical_record

logger = get_logger("MultimodalPredict")


def get_risk_level(score: int) -> str:
    """Maps 0-100 risk score to categorical triage tier."""
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Moderate"
    elif score <= 80:
        return "High"
    else:
        return "Critical"


def evaluate_safety_overrides(clinical_record: Dict[str, Any]) -> List[str]:
    """Identifies critical vital sign deviations requiring emergency override."""
    alerts = []
    temp = float(clinical_record.get("body_temperature_c", clinical_record.get("temperature", 38.6)))
    if temp >= 41.0:
        alerts.append(f"Hyperpyrexia alert: Critical body temperature ({temp:.1f}°C)")
    elif temp < 37.0:
        alerts.append(f"Hypothermia alert: Subnormal body temperature ({temp:.1f}°C)")

    activity = str(clinical_record.get("activity_level", clinical_record.get("activity", ""))).lower()
    if "recumbent" in activity or "unable_to_stand" in activity:
        alerts.append("Severe mobility impairment / Animal is recumbent")

    symptoms_str = str(clinical_record.get("symptoms", "")).lower()
    if "respiratory_distress" in symptoms_str or "severe_dyspnea" in symptoms_str:
        alerts.append("Acute respiratory distress detected")
    if "seizure" in symptoms_str or "convulsion" in symptoms_str:
        alerts.append("Neurological symptom: Seizures/Tremors")
    if "bleeding" in symptoms_str or "hemorrhage" in symptoms_str:
        alerts.append("Uncontrolled bleeding or hemorrhage")

    return alerts


def calculate_multimodal_fusion(
    image_result: Optional[Dict[str, Any]],
    clinical_result: Optional[Dict[str, Any]],
    clinical_record: Optional[Dict[str, Any]] = None,
    image_weight: float = 0.50,
    clinical_weight: float = 0.50,
) -> Dict[str, Any]:
    """Fuses visual, physiological, and anomaly signals into an interpretable screening result."""
    all_diseases = set()
    img_probs = {}
    clin_probs = {}

    if image_result and "top_predictions" in image_result:
        for p in image_result["top_predictions"]:
            img_probs[p["class"]] = p["probability"]
            all_diseases.add(p["class"])

    if clinical_result and "disease_probabilities" in clinical_result:
        clin_probs = clinical_result["disease_probabilities"]
        for d in clin_probs:
            all_diseases.add(d)

    # Compute fused probabilities
    fused_scores = {}
    for d in all_diseases:
        ip = img_probs.get(d, 0.0)
        cp = clin_probs.get(d, 0.0)

        if image_result and clinical_result:
            score = (ip * image_weight) + (cp * clinical_weight)
        elif image_result:
            score = ip
        else:
            score = cp
        fused_scores[d] = score

    # Normalize fused scores
    total_score = sum(fused_scores.values()) or 1.0
    top_disease_probs = {k: round(v / total_score, 4) for k, v in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

    top_disease = list(top_disease_probs.keys())[0] if top_disease_probs else "Unknown"
    top_fused_prob = top_disease_probs[top_disease] if top_disease_probs else 0.0

    # Model Agreement Evaluation
    img_top = image_result.get("predicted_disease") if image_result else None
    clin_top = clinical_result.get("predicted_condition") if clinical_result else None

    if img_top and clin_top:
        if img_top == clin_top:
            model_agreement = "High"
        elif img_top in clin_probs and clin_probs[img_top] > 0.25:
            model_agreement = "Moderate"
        else:
            model_agreement = "Low / Conflicting"
    else:
        model_agreement = "Single Modality Only"

    # Anomaly Status
    unknown_pattern = bool(image_result.get("unknown_pattern", False)) if image_result else False
    anomaly_score = float(image_result.get("anomaly_score", 0.0)) if image_result else 0.0

    # Risk Score Calculation (0-100)
    is_healthy_pred = top_disease.lower() in ["healthy_cattle", "healthy", "normal"]
    
    if is_healthy_pred and not unknown_pattern:
        base_risk = 15
    else:
        base_risk = int(top_fused_prob * 75)

    if unknown_pattern:
        base_risk = max(base_risk, 65)  # Anomaly increases clinical urgency while reducing certainty

    # Disagreement penalty
    if model_agreement == "Low / Conflicting":
        base_risk += 15

    # Safety Overrides Check
    safety_alerts = evaluate_safety_overrides(clinical_record) if clinical_record else []
    if safety_alerts:
        risk_score = max(85, min(100, base_risk + 30))
        risk_level = "Critical"
    else:
        risk_score = min(100, max(0, base_risk))
        risk_level = get_risk_level(risk_score)

    # Contributing Factors Synthesis
    contributing_factors = []
    if image_result and not image_result.get("is_uncertain") and not unknown_pattern:
        contributing_factors.append(f"Image visual pattern associated with {img_top} ({image_result.get('confidence', 0)*100:.1f}% confidence)")
    elif unknown_pattern:
        contributing_factors.append("Image visual features deviate from known training patterns (potential anomaly/unseen condition)")

    if clinical_record:
        temp = float(clinical_record.get("body_temperature_c", clinical_record.get("temperature", 38.6)))
        if temp > 39.5:
            contributing_factors.append(f"Elevated body temperature ({temp:.1f}°C)")
        symptoms = str(clinical_record.get("symptoms", ""))
        if symptoms and symptoms != "none":
            contributing_factors.append(f"Reported symptoms: {symptoms.replace('_', ' ')}")
        vacc = str(clinical_record.get("vaccination_status", ""))
        if vacc.lower() in ["overdue", "unvaccinated"]:
            contributing_factors.append(f"Vaccination status: {vacc}")

    for alert in safety_alerts:
        contributing_factors.append(f"CRITICAL OBSERVATION: {alert}")

    # Recommended Action
    if safety_alerts or risk_level == "Critical":
        recommended_action = "Urgent veterinary emergency assessment required"
        vet_referral = True
        lab_confirmation = "Urgent laboratory confirmation required"
    elif unknown_pattern:
        recommended_action = "Veterinary examination and laboratory diagnostic panel recommended (Unknown pattern)"
        vet_referral = True
        lab_confirmation = "Recommended (PCR / Serology / Culture)"
    elif model_agreement == "Low / Conflicting":
        recommended_action = "Veterinary clinical assessment recommended to resolve conflicting screening signals"
        vet_referral = True
        lab_confirmation = "Recommended where clinically indicated"
    elif risk_level == "High":
        recommended_action = "Veterinary consultation recommended"
        vet_referral = True
        lab_confirmation = "Recommended"
    elif risk_level == "Moderate":
        recommended_action = "Monitor animal closely; consult veterinarian if symptoms persist or escalate"
        vet_referral = True
        lab_confirmation = "Consider if symptoms do not resolve"
    else:
        recommended_action = "Routine herd health monitoring"
        vet_referral = False
        lab_confirmation = "Not immediately indicated"

    screening_name = f"Possible {top_disease}" if not is_healthy_pred else "Healthy / No Active Lesions Detected"
    if unknown_pattern:
        screening_name = "Unknown / Unusual Pattern (Atypical Disease Presentation)"

    output = {
        "screening_prediction": screening_name,
        "top_disease_probabilities": top_disease_probs,
        "image_confidence": image_result.get("confidence") if image_result else None,
        "clinical_probability": clinical_result.get("clinical_probability") if clinical_result else None,
        "unknown_pattern": unknown_pattern,
        "anomaly_score": anomaly_score,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model_agreement": model_agreement,
        "safety_overrides_triggered": len(safety_alerts) > 0,
        "safety_alerts": safety_alerts,
        "contributing_factors": contributing_factors,
        "recommended_action": recommended_action,
        "vet_referral": vet_referral,
        "lab_confirmation": lab_confirmation,
        "disclaimer": "This is an AI screening and decision-support result and not a confirmed veterinary diagnosis.",
    }

    return output


def run_multimodal_prediction(
    image_path: Optional[str] = None,
    clinical_record: Optional[Dict[str, Any]] = None,
    image_model_path: str = "models/image_model/best_model.pth",
    clinical_model_path: str = "models/clinical_model/model.joblib",
    clinical_prep_path: str = "models/clinical_model/preprocessor.joblib",
    anomaly_path: str = "models/anomaly_model/image_anomaly.joblib",
) -> Dict[str, Any]:
    """Coordinates image and clinical inference and executes multimodal fusion."""
    img_res = None
    if image_path and os.path.exists(image_path) and os.path.exists(image_model_path):
        img_res = predict_image(
            image_path=image_path,
            model_path=image_model_path,
            anomaly_path=anomaly_path,
        )

    clin_res = None
    if clinical_record and os.path.exists(clinical_model_path) and os.path.exists(clinical_prep_path):
        clin_res = predict_clinical(
            record_dict=clinical_record,
            model_path=clinical_model_path,
            preprocessor_path=clinical_prep_path,
        )

    if not img_res and not clin_res:
        raise ValueError("At least one valid image or clinical input must be provided with trained models.")

    fused_res = calculate_multimodal_fusion(
        image_result=img_res,
        clinical_result=clin_res,
        clinical_record=clinical_record,
    )

    return fused_res


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Multimodal Prediction")
    parser.add_argument("--image", type=str, default=None, help="Path to lesion image")
    parser.add_argument("--age", type=float, default=4.0, help="Age in years")
    parser.add_argument("--breed", type=str, default="Gir", help="Breed")
    parser.add_argument("--sex", type=str, default="Female", help="Sex")
    parser.add_argument("--weight", type=float, default=430.0, help="Weight in kg")
    parser.add_argument("--temperature", type=float, default=40.4, help="Body temperature in Celsius")
    parser.add_argument("--heart-rate", type=int, default=86, help="Heart rate bpm")
    parser.add_argument("--respiratory-rate", type=int, default=36, help="Respiratory rate bpm")
    parser.add_argument("--milk-drop", type=float, default=50.0, help="Milk drop pct")
    parser.add_argument("--appetite", type=str, default="Anorexia", help="Appetite")
    parser.add_argument("--activity", type=str, default="Lethargic", help="Activity")
    parser.add_argument("--vaccination", type=str, default="Overdue", help="Vaccination status")
    parser.add_argument("--symptoms", type=str, default="fever,skin_nodules,loss_of_appetite", help="Symptoms")
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

    result = run_multimodal_prediction(
        image_path=args.image,
        clinical_record=record,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
