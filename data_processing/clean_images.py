"""Image Data Cleaning Engine for PashuRaksha AI.

Performs non-destructive cleaning by identifying corrupted images, zero-byte files,
unsupported formats, and duplicate images via SHA-256 hashing.
Exports audit CSVs and copies clean images to data/processed/images/.
"""

import os
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger("CleanImages")


def clean_image_dataset(
    source_dir: str = "data/images",
    dest_dir: str = "data/processed/images",
    reports_dir: str = "reports",
) -> Tuple[int, int, int]:
    """Cleans images non-destructively and prepares the processed directory."""
    src_path = Path(source_dir)
    dst_path = Path(dest_dir)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 0, 0, 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = [d.name for d in src_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    corrupted_records = []
    duplicate_records = []
    file_hashes = {}
    total_processed = 0
    total_copied = 0

    logger.info(f"Scanning and cleaning images from: {source_dir}")

    for c in sorted(classes):
        class_src_dir = src_path / c
        class_dst_dir = dst_path / c
        class_dst_dir.mkdir(parents=True, exist_ok=True)

        for f in class_src_dir.iterdir():
            if not f.is_file() or f.name.startswith("."):
                continue

            total_processed += 1
            ext = f.suffix.lower()

            # 1. Format check
            if ext not in valid_extensions:
                corrupted_records.append({
                    "file_path": str(f),
                    "class": c,
                    "reason": f"Unsupported file extension: {ext}",
                })
                continue

            # 2. Zero-byte check
            if f.stat().st_size == 0:
                corrupted_records.append({
                    "file_path": str(f),
                    "class": c,
                    "reason": "Zero-byte file",
                })
                continue

            # 3. Readability & Header check
            try:
                with Image.open(f) as img:
                    img.verify()
                # Confirm we can convert and load pixel data
                with Image.open(f) as img:
                    img.convert("RGB").load()
            except Exception as e:
                corrupted_records.append({
                    "file_path": str(f),
                    "class": c,
                    "reason": f"Corrupted image data: {str(e)}",
                })
                continue

            # 4. Duplicate check via SHA-256
            with open(f, "rb") as bf:
                fhash = hashlib.sha256(bf.read()).hexdigest()

            if fhash in file_hashes:
                duplicate_records.append({
                    "duplicate_path": str(f),
                    "original_path": file_hashes[fhash]["path"],
                    "class": c,
                    "hash": fhash,
                })
                # Skip copying duplicate to prevent training leakage
                continue

            file_hashes[fhash] = {"path": str(f), "class": c}

            # Copy clean image to destination
            target_file = class_dst_dir / f.name
            shutil.copy2(f, target_file)
            total_copied += 1

    # Save reports
    df_corrupted = pd.DataFrame(corrupted_records)
    df_duplicates = pd.DataFrame(duplicate_records)

    corrupted_csv = rep_path / "corrupted_images.csv"
    duplicate_csv = rep_path / "duplicate_images.csv"

    df_corrupted.to_csv(corrupted_csv, index=False)
    df_duplicates.to_csv(duplicate_csv, index=False)

    logger.info(f"Cleaned images copied to: {dest_dir}")
    logger.info(f"Total scanned: {total_processed} | Clean copied: {total_copied} | Corrupted: {len(corrupted_records)} | Duplicates: {len(duplicate_records)}")
    logger.info(f"Reports saved to: {corrupted_csv} and {duplicate_csv}")

    return total_processed, total_copied, len(corrupted_records)


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Image Cleaning")
    parser.add_argument("--source-dir", type=str, default="data/images", help="Raw images directory")
    parser.add_argument("--dest-dir", type=str, default="data/processed/images", help="Clean images destination")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports destination folder")
    args = parser.parse_args()

    clean_image_dataset(args.source_dir, args.dest_dir, args.reports_dir)


if __name__ == "__main__":
    main()
