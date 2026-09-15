"""Offline Image Inference Engine for PashuRaksha AI.

Performs calibrated disease prediction on livestock lesion images,
extracts embeddings, checks anomaly status, and returns top-K probabilities.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import joblib

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from training.train_image import build_mobilenet_v3, get_transforms
from training.train_anomaly import FeatureExtractor

logger = get_logger("ImagePredict")


def predict_image(
    image_path: str,
    model_path: str = "models/image_model/best_model.pth",
    anomaly_path: str = "models/anomaly_model/image_anomaly.joblib",
    anomaly_config_path: str = "models/anomaly_model/anomaly_config.json",
    confidence_threshold: float = 0.60,
    top_k: int = 3,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Inference for single livestock image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Image model not found: {model_path}")

    # Select device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["class_names"]
    num_classes = len(classes)
    temperature = checkpoint.get("temperature", 1.0)

    model = build_mobilenet_v3(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load and transform image
    _, eval_transform = get_transforms()
    with Image.open(image_path) as raw_img:
        img_rgb = raw_img.convert("RGB")
    tensor_img = eval_transform(img_rgb).unsqueeze(0).to(device)

    # 1. Image Classification with Temperature Scaling
    with torch.no_grad():
        logits = model(tensor_img)
        calibrated_probs = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()

    # Top-K predictions
    sorted_indices = np.argsort(calibrated_probs)[::-1]
    top_predictions = []
    for idx in sorted_indices[:top_k]:
        top_predictions.append({
            "class": classes[idx],
            "probability": round(float(calibrated_probs[idx]), 4),
        })

    top_class = top_predictions[0]["class"]
    top_confidence = top_predictions[0]["probability"]

    # 2. Extract Embedding and Check Anomaly Status
    anomaly_score = 0.0
    unknown_pattern = False
    anomaly_threshold = 0.50

    if os.path.exists(anomaly_path):
        iso_forest = joblib.load(anomaly_path)
        extractor = FeatureExtractor(model).to(device)
        extractor.eval()
        with torch.no_grad():
            emb = extractor(tensor_img).cpu().numpy()
        # Negative anomaly score
        anomaly_score = float(-iso_forest.score_samples(emb)[0])

        if os.path.exists(anomaly_config_path):
            with open(anomaly_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                anomaly_threshold = cfg.get("anomaly_threshold", 0.50)

        if anomaly_score > anomaly_threshold:
            unknown_pattern = True

    # 3. Low-Confidence / Unknown Pattern Decision Logic
    is_uncertain = top_confidence < confidence_threshold

    if unknown_pattern:
        screening_prediction = "Unknown / Unusual Pattern"
        screening_note = (
            "Visual features deviate significantly from known training distributions. "
            "Veterinary examination and/or laboratory testing is recommended."
        )
    elif is_uncertain:
        screening_prediction = f"Uncertain (Possible {top_class})"
        screening_note = (
            f"Prediction confidence ({top_confidence:.2f}) is below prototype decision threshold ({confidence_threshold:.2f}). "
            "Veterinary review recommended."
        )
    else:
        screening_prediction = top_class
        screening_note = "AI screening result. Veterinary examination and laboratory confirmation recommended where clinically indicated."

    result = {
        "image_file": str(Path(image_path).name),
        "predicted_disease": screening_prediction,
        "confidence": top_confidence,
        "is_uncertain": is_uncertain,
        "unknown_pattern": unknown_pattern,
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_threshold": round(anomaly_threshold, 4),
        "top_predictions": top_predictions,
        "screening_note": screening_note,
        "disclaimer": "This is an AI decision-support screening prototype and not a definitive veterinary diagnosis.",
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="models/image_model/best_model.pth", help="Model checkpoint")
    parser.add_argument("--anomaly", type=str, default="models/anomaly_model/image_anomaly.joblib", help="Anomaly model")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Top K predictions")
    args = parser.parse_args()

    res = predict_image(
        image_path=args.image,
        model_path=args.model,
        anomaly_path=args.anomaly,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
