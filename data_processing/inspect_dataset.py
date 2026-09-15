"""Dataset Inspection Engine for PashuRaksha AI.

Inspects image and clinical/tabular datasets, computes class distributions,
checks for corruption, duplicates, and missing values, and generates
visualizations and structured JSON reports.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger("InspectDataset")


def inspect_image_dataset(data_dir: str) -> Dict[str, Any]:
    """Scans and analyzes an image dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Image dataset path not found: {data_dir}")
        return {"error": f"Path not found: {data_dir}", "total_images": 0}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = [d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    total_images = 0
    class_counts = {}
    formats = Counter()
    resolutions = []
    corrupted_files = []
    zero_byte_files = []
    unsupported_files = []
    file_hashes = {}
    duplicate_files = []

    for c in sorted(classes):
        class_dir = data_path / c
        img_files = [f for f in class_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        valid_in_class = 0

        for f in img_files:
            ext = f.suffix.lower()
            if ext not in valid_extensions:
                unsupported_files.append(str(f))
                continue

            if f.stat().st_size == 0:
                zero_byte_files.append(str(f))
                continue

            # Read and verify image integrity
            try:
                with Image.open(f) as img:
                    img.verify()
                # Re-open for resolution check (verify closes file stream)
                with Image.open(f) as img:
                    formats[img.format or ext.replace(".", "").upper()] += 1
                    resolutions.append(img.size)  # (width, height)
                
                # Check duplicates via hash
                with open(f, "rb") as bf:
                    fhash = hashlib.sha256(bf.read()).hexdigest()
                if fhash in file_hashes:
                    duplicate_files.append({"original": file_hashes[fhash], "duplicate": str(f)})
                else:
                    file_hashes[fhash] = str(f)

                valid_in_class += 1
                total_images += 1
            except Exception as e:
                corrupted_files.append({"path": str(f), "error": str(e)})

        class_counts[c] = valid_in_class

    # Imbalance ratio
    counts_list = list(class_counts.values()) if class_counts else [0]
    min_samples = min(counts_list) if counts_list and min(counts_list) > 0 else 0
    max_samples = max(counts_list) if counts_list else 0
    imbalance_ratio = round(max_samples / max(1, min_samples), 2) if min_samples > 0 else 0.0

    res_stats = {}
    if resolutions:
        widths, heights = zip(*resolutions)
        res_stats = {
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "mean_width": float(np.mean(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "mean_height": float(np.mean(heights)),
        }

    report = {
        "dataset_directory": str(data_path.resolve()),
        "total_valid_images": total_images,
        "num_classes": len(classes),
        "classes": sorted(classes),
        "class_distribution": class_counts,
        "min_samples_per_class": min_samples,
        "max_samples_per_class": max_samples,
        "class_imbalance_ratio": imbalance_ratio,
        "formats": dict(formats),
        "resolution_stats": res_stats,
        "corrupted_images_count": len(corrupted_files),
        "corrupted_images": corrupted_files,
        "duplicate_images_count": len(duplicate_files),
        "duplicate_images": duplicate_files,
        "zero_byte_files_count": len(zero_byte_files),
        "unsupported_files_count": len(unsupported_files),
    }

    return report


def inspect_clinical_dataset(file_path: str) -> Dict[str, Any]:
    """Inspects tabular clinical data file."""
    p = Path(file_path)
    if not p.exists():
        logger.warning(f"Clinical dataset not found: {file_path}")
        return {"error": f"Path not found: {file_path}", "total_rows": 0}

    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        elif p.suffix == ".json":
            df = pd.read_json(p)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            return {"error": f"Unsupported clinical format: {p.suffix}"}
    except Exception as e:
        return {"error": f"Failed to read clinical dataset: {str(e)}"}

    target_col = None
    candidate_targets = ["disease", "target", "diagnosis", "condition", "risk_level", "health_status", "label"]
    for col in df.columns:
        if col.lower() in candidate_targets:
            target_col = col
            break

    target_dist = {}
    if target_col:
        target_dist = df[target_col].value_counts().to_dict()

    missing_vals = df.isnull().sum().to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    duplicates = int(df.duplicated().sum())

    report = {
        "file_path": str(p.resolve()),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "data_types": dtypes,
        "missing_values": missing_vals,
        "duplicate_rows": duplicates,
        "target_column": target_col,
        "target_distribution": target_dist,
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }
    return report


def plot_class_distribution(class_distribution: Dict[str, int], output_path: str) -> None:
    """Generates and saves a clean class distribution bar plot."""
    if not class_distribution:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, counts, color="#2b5c8f", edgecolor="#1a3b5c", alpha=0.85)
    plt.title("Livestock Disease Dataset Class Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Disease Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Sample Count", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(1, max(counts) * 0.01),
            int(yval),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Class distribution graph saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Dataset Inspector")
    parser.add_argument("--image-dir", type=str, default="data/images", help="Path to image dataset")
    parser.add_argument("--clinical-file", type=str, default="data/clinical/clinical_data.csv", help="Path to clinical tabular file")
    parser.add_argument("--output-report", type=str, default="reports/dataset_report.json", help="Path to save report JSON")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    
    logger.info("==========================================")
    logger.info("   PASHURAKSHA AI - DATASET INSPECTION    ")
    logger.info("==========================================")

    # Inspect images
    img_report = inspect_image_dataset(args.image_dir)
    
    # Inspect clinical
    clin_report = inspect_clinical_dataset(args.clinical_file)

    full_report = {
        "image_dataset": img_report,
        "clinical_dataset": clin_report,
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Inspection report saved to: {args.output_report}")

    # Plot image distribution if images present
    if img_report.get("class_distribution"):
        plot_path = "reports/image_class_distribution.png"
        plot_class_distribution(img_report["class_distribution"], plot_path)

    # Print summary
    print("\n" + "="*50)
    print("           DATASET SUMMARY REPORT          ")
    print("="*50)
    print(f"Total Valid Images      : {img_report.get('total_valid_images', 0)}")
    print(f"Disease Classes         : {img_report.get('num_classes', 0)}")
    print("Class Distribution      :")
    for cls, cnt in img_report.get("class_distribution", {}).items():
        print(f"  - {cls:22s}: {cnt} samples")
    print(f"Corrupted Images        : {img_report.get('corrupted_images_count', 0)}")
    print(f"Duplicate Images        : {img_report.get('duplicate_images_count', 0)}")
    print(f"Class Imbalance Ratio   : {img_report.get('class_imbalance_ratio', 0.0)}")
    print("-" * 50)
    print(f"Clinical Dataset Rows   : {clin_report.get('total_rows', 0)}")
    print(f"Clinical Columns        : {clin_report.get('total_columns', 0)}")
    print(f"Clinical Target Column  : {clin_report.get('target_column', 'None')}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
