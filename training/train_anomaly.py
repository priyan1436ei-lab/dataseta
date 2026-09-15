"""Out-of-Distribution (OOD) & Anomaly Detection Pipeline for PashuRaksha AI.

Extracts deep feature embeddings from MobileNetV3 and fits an Isolation Forest model
to detect rare, unusual, or unseen disease patterns.
Saves:
  - models/anomaly_model/image_anomaly.joblib
  - models/anomaly_model/anomaly_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything
from training.train_image import build_mobilenet_v3, get_transforms, LivestockDataset

logger = get_logger("TrainAnomaly")


class FeatureExtractor(nn.Module):
    """Extracts 960-dim latent feature embeddings from MobileNetV3."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extracts feature embedding vectors from all batches."""
    extractor = FeatureExtractor(model).to(device)
    extractor.eval()
    embeddings = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            emb = extractor(images)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def train_anomaly_detector(
    image_model_path: str = "models/image_model/best_model.pth",
    split_csv: str = "reports/dataset_split.csv",
    output_dir: str = "models/anomaly_model",
    contamination: float = 0.05,
    n_estimators: int = 150,
    seed: int = 42,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Extracts training image embeddings and fits Isolation Forest."""
    seed_everything(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(image_model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {image_model_path}")
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

    # Load image model
    checkpoint = torch.load(image_model_path, map_location=device)
    classes = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]

    model = build_mobilenet_v3(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load splits
    df_split = pd.read_csv(split_csv)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()

    _, eval_transform = get_transforms()
    train_dataset = LivestockDataset(train_df, transform=eval_transform, class_to_idx=class_to_idx)
    val_dataset = LivestockDataset(val_df, transform=eval_transform, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    logger.info("Extracting feature embeddings from MobileNetV3 backbone...")
    train_embeddings = extract_embeddings(model, train_loader, device)
    val_embeddings = extract_embeddings(model, val_loader, device)
    logger.info(f"Extracted {train_embeddings.shape[0]} training embeddings of dimension {train_embeddings.shape[1]}.")

    # Fit Isolation Forest on normal training embeddings
    logger.info(f"Fitting Isolation Forest (contamination={contamination}, n_estimators={n_estimators})...")
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    iso_forest.fit(train_embeddings)

    # Compute raw anomaly scores on validation embeddings
    # Note: score_samples returns negative anomaly score (lower means more anomalous)
    val_scores = -iso_forest.score_samples(val_embeddings)
    score_threshold = float(np.percentile(val_scores, 95.0))  # 95th percentile as prototype anomaly boundary

    logger.info(f"Validation anomaly scores: Min={val_scores.min():.4f}, Max={val_scores.max():.4f}, Mean={val_scores.mean():.4f}")
    logger.info(f"Computed 95th percentile anomaly threshold: {score_threshold:.4f}")

    # Save detector & config
    detector_path = Path(output_dir) / "image_anomaly.joblib"
    joblib.dump(iso_forest, detector_path)

    config = {
        "model_type": "IsolationForest",
        "embedding_dim": int(train_embeddings.shape[1]),
        "anomaly_threshold": score_threshold,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "validation_stats": {
            "min_score": float(val_scores.min()),
            "max_score": float(val_scores.max()),
            "mean_score": float(val_scores.mean()),
            "threshold_95pct": score_threshold,
        },
        "disclaimer": "Threshold is a prototype decision boundary. High anomaly scores indicate out-of-distribution patterns requiring veterinary examination.",
    }

    config_path = Path(output_dir) / "anomaly_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved anomaly detector model to: {detector_path}")
    logger.info(f"Saved anomaly config to: {config_path}")

    return config


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Anomaly Detection Training")
    parser.add_argument("--image-model", type=str, default="models/image_model/best_model.pth", help="Trained image model")
    parser.add_argument("--split-csv", type=str, default="reports/dataset_split.csv", help="Split CSV")
    parser.add_argument("--output-dir", type=str, default="models/anomaly_model", help="Output directory")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected outlier ratio")
    parser.add_argument("--n-estimators", type=int, default=150, help="Isolation forest trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    train_anomaly_detector(
        image_model_path=args.image_model,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
