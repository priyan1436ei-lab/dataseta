"""Dataset Splitting Pipeline for PashuRaksha AI.

Generates stratified 70% Training / 15% Validation / 15% Testing splits,
guaranteeing zero image leakage across partitions.
Outputs reports/dataset_split.csv.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything

logger = get_logger("PrepareImages")


def split_image_dataset(
    processed_dir: str = "data/processed/images",
    output_split_csv: str = "reports/dataset_split.csv",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Performs stratified split and saves split assignment."""
    seed_everything(seed)
    proc_path = Path(processed_dir)
    if not proc_path.exists():
        logger.error(f"Processed images directory not found: {processed_dir}")
        return pd.DataFrame()

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []

    classes = [d.name for d in proc_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for c in sorted(classes):
        class_dir = proc_path / c
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions:
                records.append({
                    "file_path": str(f.resolve()),
                    "relative_path": str(f.relative_to(proc_path)),
                    "class": c,
                })

    df = pd.DataFrame(records)
    if df.empty:
        logger.error("No images found to split!")
        return df

    # Check minimum class size for stratification
    min_class_count = df["class"].value_counts().min()
    stratify_col = df["class"] if min_class_count >= 3 else None

    # Step 1: Split train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=stratify_col,
    )

    # Step 2: Split temp into val and test (50/50 of the remaining 30%)
    temp_stratify = temp_df["class"] if (stratify_col is not None and temp_df["class"].value_counts().min() >= 2) else None
    val_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=temp_stratify,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    os.makedirs(os.path.dirname(output_split_csv), exist_ok=True)
    split_df.to_csv(output_split_csv, index=False)

    logger.info("==========================================")
    logger.info(f"Dataset Split Completed (Seed={seed}):")
    logger.info(f"  - Train samples : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Val samples   : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Test samples  : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Total samples : {len(df)}")
    logger.info(f"Saved dataset split assignment to: {output_split_csv}")
    logger.info("==========================================")

    return split_df


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Stratified Dataset Splitter")
    parser.add_argument("--processed-dir", type=str, default="data/processed/images", help="Path to clean processed images")
    parser.add_argument("--output-csv", type=str, default="reports/dataset_split.csv", help="Output split CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_image_dataset(args.processed_dir, args.output_csv, args.seed)


if __name__ == "__main__":
    main()
