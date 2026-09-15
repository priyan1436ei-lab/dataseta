"""Explainable AI (XAI) for PashuRaksha AI Image Classifier via Grad-CAM.

Generates visual Class Activation Maps highlighting regions influencing the model's
livestock disease prediction.
Includes veterinary disclaimer noting that heatmaps reflect statistical network attention
rather than definitive medical causation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from training.train_image import build_mobilenet_v3, get_transforms

logger = get_logger("ImageExplain")


class GradCAM:
    """Gradient-weighted Class Activation Mapping for MobileNetV3."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generates 2D normalized Grad-CAM heatmap (0.0 to 1.0)."""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # Backward pass for target class
        score = logits[0, class_idx]
        score.backward()

        # Compute weights via Global Average Pooling of gradients
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0).detach().cpu().numpy()

        # Apply ReLU to retain only positive influences
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam


def explain_image(
    image_path: str,
    checkpoint_path: str = "models/image_model/best_model.pth",
    output_dir: str = "reports/explanations/images",
    device_str: str = "auto",
) -> Tuple[str, float, str]:
    """Generates Grad-CAM visual explanation overlay and saves figure."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["class_names"]
    num_classes = len(classes)

    model = build_mobilenet_v3(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Target last conv layer in MobileNetV3 features
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # Load and transform image
    _, eval_transform = get_transforms()
    raw_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(raw_img.resize((224, 224)))

    tensor_img = eval_transform(raw_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

    # Compute CAM
    tensor_img_grad = tensor_img.clone().requires_grad_(True)
    cam = grad_cam.generate(tensor_img_grad, class_idx=pred_idx)

    # Resize CAM to image dimensions
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    overlay = np.uint8(0.6 * orig_np + 0.4 * heatmap)

    # Create figure with side-by-side view and disclaimer
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(orig_np)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM Attention Map\n({pred_class})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay Analysis\nConfidence: {confidence*100:.1f}%", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        "PashuRaksha AI — Visual Explainability (Grad-CAM)\n"
        "[Notice: Highlights indicate network feature activation, not definitive pathological confirmation.]",
        fontsize=10,
        fontstyle="italic",
        y=0.98,
    )

    base_name = Path(image_path).stem
    out_file = Path(output_dir) / f"{base_name}_gradcam.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated Grad-CAM explanation for {pred_class} ({confidence*100:.1f}%): {out_file}")
    return pred_class, confidence, str(out_file)


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Grad-CAM Visual Explainability")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/image_model/best_model.pth", help="Model checkpoint")
    parser.add_argument("--output-dir", type=str, default="reports/explanations/images", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    explain_image(args.image, args.model, args.output_dir, args.device)


if __name__ == "__main__":
    main()
