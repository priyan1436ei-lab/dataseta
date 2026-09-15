"""MobileNetV3 Livestock Disease Image Training Pipeline for PashuRaksha AI.

Features:
- MobileNetV3-Large with ImageNet pretrained weights.
- 2-Stage Transfer Learning:
    Stage 1: Frozen backbone, train classification head.
    Stage 2: Fine-tune deeper backbone layers with lower learning rate.
- Class-weighted CrossEntropyLoss for imbalance handling.
- Early stopping based on validation macro-F1 / loss.
- Post-hoc Temperature Scaling probability calibration.
- Saves checkpoint models/image_model/best_model.pth.
- Generates loss, accuracy, F1, and learning rate curves in reports/image_training/.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything
from utils.metrics import compute_classification_metrics

logger = get_logger("TrainImage")


# ---------------------------------------------------------------------------
# Dataset & Transforms
# ---------------------------------------------------------------------------
class LivestockDataset(Dataset):
    """Custom PyTorch dataset for livestock health images."""

    def __init__(self, df: pd.DataFrame, transform=None, class_to_idx: Dict[str, int] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(sorted(df["class"].unique()))}
        self.classes = sorted(list(self.class_to_idx.keys()))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        target = self.class_to_idx[row["class"]]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test image transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_mobilenet_v3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV3-Large with replaced classifier head."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # In MobileNetV3-Large, classifier is a Sequential block
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Temperature Scaling for Calibration
# ---------------------------------------------------------------------------
class ModelWithTemperature(nn.Module):
    """Calibrates model logits with learned temperature scaling."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(self, valid_loader: DataLoader, device: torch.device):
        """Tunes temperature parameter using NLL on validation set."""
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        logger.info(f"Learned optimal temperature: {self.temperature.item():.4f}")


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def train_model(
    data_dir: str = "data/processed/images",
    split_csv: str = "reports/dataset_split.csv",
    output_dir: str = "models/image_model",
    reports_dir: str = "reports/image_training",
    epochs_stage1: int = 15,
    epochs_stage2: int = 25,
    batch_size: int = 16,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Executes 2-stage transfer learning with early stopping and calibration."""
    seed_everything(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Selected compute device: {device}")

    # Load dataset split
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df_split = pd.read_csv(split_csv)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()

    classes = sorted(train_df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    logger.info(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples across {num_classes} classes.")

    train_transform, eval_transform = get_transforms()
    train_dataset = LivestockDataset(train_df, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = LivestockDataset(val_df, transform=eval_transform, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute class weights for imbalanced loss
    class_counts = train_df["class"].value_counts().to_dict()
    total_samples = len(train_df)
    class_weights = torch.tensor(
        [total_samples / (num_classes * max(1, class_counts.get(c, 1))) for c in classes],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_mobilenet_v3(num_classes=num_classes, pretrained=True).to(device)

    # Metrics history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_macro_f1 = 0.0
    patience_counter = 0
    best_state_dict = None
    best_epoch = 0

    # ---------------------------------------------------------
    # STAGE 1: Train classification head only
    # ---------------------------------------------------------
    logger.info(">>> STARTING STAGE 1: Training Classification Head (Backbone Frozen) <<<")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr_stage1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage1)

    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss, v_preds, v_targets = 0.0, [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_preds.extend(preds.cpu().numpy())
                v_targets.extend(targets.cpu().numpy())

        val_loss = v_loss / max(1, len(val_dataset))
        val_metrics = compute_classification_metrics(np.array(v_targets), np.array(v_preds), class_names=classes)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(f"Stage 1 Epoch [{epoch:02d}/{epochs_stage1:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Macro-F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            best_epoch = epoch

    # ---------------------------------------------------------
    # STAGE 2: Fine-tuning deeper layers
    # ---------------------------------------------------------
    logger.info(">>> STARTING STAGE 2: Fine-Tuning Deeper Layers <<<")
    # Unfreeze top layers of feature extractor
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.features[-4:].parameters(), "lr": lr_stage2 * 0.5},
        {"params": model.classifier.parameters(), "lr": lr_stage2},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    for epoch in range(1, epochs_stage2 + 1):
        global_epoch = epochs_stage1 + epoch
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += images.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss, v_preds, v_targets = 0.0, [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_preds.extend(preds.cpu().numpy())
                v_targets.extend(targets.cpu().numpy())

        val_loss = v_loss / max(1, len(val_dataset))
        val_metrics = compute_classification_metrics(np.array(v_targets), np.array(v_preds), class_names=classes)
        scheduler.step(val_metrics["macro_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(f"Stage 2 Epoch [{epoch:02d}/{epochs_stage2:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Macro-F1: {val_metrics['macro_f1']:.4f}")

        # Early stopping check
        if val_metrics["macro_f1"] > best_macro_f1 or (val_metrics["macro_f1"] == best_macro_f1 and val_loss < best_val_loss):
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            best_epoch = global_epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at global epoch {global_epoch} (Patience={patience}).")
                break

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---------------------------------------------------------
    # Temperature Scaling Calibration
    # ---------------------------------------------------------
    logger.info("Calibrating model probability outputs via Temperature Scaling...")
    calibrated_model = ModelWithTemperature(model)
    calibrated_model.calibrate(val_loader, device)
    learned_temperature = float(calibrated_model.temperature.item())

    # Save Checkpoint
    checkpoint_path = Path(output_dir) / "best_model.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": classes,
        "class_to_idx": class_to_idx,
        "model_name": "mobilenet_v3_large",
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "temperature": learned_temperature,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_loss": best_val_loss,
        "training_config": {
            "stage1_epochs": epochs_stage1,
            "stage2_epochs": epochs_stage2,
            "batch_size": batch_size,
            "lr_stage1": lr_stage1,
            "lr_stage2": lr_stage2,
            "seed": seed,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved best image model checkpoint to: {checkpoint_path}")

    # Plot and Save Curves
    _plot_training_curves(history, reports_dir)

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_loss": best_val_loss,
        "temperature": learned_temperature,
    }


def _plot_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    """Renders training and validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    plt.title("Image Classifier: Training & Validation Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, history["val_acc"], "g--", label="Val Accuracy", linewidth=2)
    plt.title("Image Classifier: Accuracy Curve", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # 3. Macro F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_f1"], "m-", label="Validation Macro F1", linewidth=2)
    plt.title("Image Classifier: Validation Macro F1 Score", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=300)
    plt.close()

    # 4. Learning Rate Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], "orange", label="Learning Rate", linewidth=2)
    plt.title("Image Classifier: Learning Rate Schedule", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate_curve.png"), dpi=300)
    plt.close()

    logger.info(f"Saved training visualization curves to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI MobileNetV3 Training")
    parser.add_argument("--data-dir", type=str, default="data/processed/images", help="Processed images directory")
    parser.add_argument("--split-csv", type=str, default="reports/dataset_split.csv", help="Dataset split CSV")
    parser.add_argument("--output-dir", type=str, default="models/image_model", help="Model checkpoint folder")
    parser.add_argument("--reports-dir", type=str, default="reports/image_training", help="Training curves folder")
    parser.add_argument("--epochs-s1", type=int, default=15, help="Stage 1 head epochs")
    parser.add_argument("--epochs-s2", type=int, default=25, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-s1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr-s2", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda)")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        epochs_stage1=args.epochs_s1,
        epochs_stage2=args.epochs_s2,
        batch_size=args.batch_size,
        lr_stage1=args.lr_s1,
        lr_stage2=args.lr_s2,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
