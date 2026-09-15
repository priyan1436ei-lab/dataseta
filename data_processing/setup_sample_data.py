"""Sample Livestock Dataset Generator for PashuRaksha AI.

Generates realistic livestock disease image datasets and clinical tabular records
representing major ruminant diseases (Lumpy Skin Disease, Foot & Mouth Disease,
Bovine Mastitis, Blackleg, and Healthy Cattle) for reproducible end-to-end testing,
validation, Grad-CAM, SHAP, and multimodal decision-support demonstration.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything

logger = get_logger("SetupSampleData")


def generate_synthetic_livestock_images(
    output_dir: str = "data/images",
    num_samples_per_class: int = 40,
    seed: int = 42,
) -> None:
    """Generates visual pattern representations of livestock disease lesions."""
    seed_everything(seed)
    np.random.seed(seed)
    out_path = Path(output_dir)

    classes = [
        "Lumpy_Skin_Disease",
        "Foot_and_Mouth_Disease",
        "Bovine_Mastitis",
        "Blackleg",
        "Healthy_Cattle",
    ]

    for c in classes:
        class_dir = out_path / c
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Create base hide/skin texture (256x256 RGB)
            base_color = np.random.randint(120, 180, size=3)
            img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
            
            # Add realistic texture noise
            noise = np.random.normal(0, 15, (256, 256, 3)).astype(np.int16)
            img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)

            if c == "Lumpy_Skin_Disease":
                # Distinct raised nodules/nodular lesions
                num_nodules = np.random.randint(6, 15)
                for _ in range(num_nodules):
                    x = np.random.randint(30, 220)
                    y = np.random.randint(30, 220)
                    r = np.random.randint(10, 25)
                    # Outer dark halo
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(80, 45, 35))
                    # Inner necrotic/raised center
                    draw.ellipse([x - r//2, y - r//2, x + r//2, y + r//2], fill=(160, 90, 75))

            elif c == "Foot_and_Mouth_Disease":
                # Blister/vesicular erosions, interdigital or oral lesions
                num_blisters = np.random.randint(3, 8)
                for _ in range(num_blisters):
                    x = np.random.randint(50, 200)
                    y = np.random.randint(50, 200)
                    rx = np.random.randint(15, 35)
                    ry = np.random.randint(8, 20)
                    draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(190, 50, 50))
                    draw.ellipse([x - rx + 3, y - ry + 3, x + rx - 3, y + ry - 3], fill=(230, 120, 110))

            elif c == "Bovine_Mastitis":
                # Swollen, inflamed, erythematous quarter tissue pattern
                cx, cy = 128 + np.random.randint(-20, 20), 128 + np.random.randint(-20, 20)
                r = np.random.randint(50, 85)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 70, 70))
                draw.ellipse([cx - r//2, cy - r//2, cx + r//2, cy + r//2], fill=(240, 110, 100))

            elif c == "Blackleg":
                # Dark, emphysematous, crepitant muscle swelling
                cx, cy = np.random.randint(80, 170), np.random.randint(80, 170)
                r = np.random.randint(40, 75)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 40, 45))
                draw.ellipse([cx - r//2, cy - r//2, cx + r//2, cy + r//2], fill=(70, 65, 60))

            elif c == "Healthy_Cattle":
                # Smooth, uniform coat pattern without focal lesions
                # Subtle coat color patch
                if np.random.rand() > 0.5:
                    draw.ellipse([50, 50, 200, 200], fill=tuple(np.clip(base_color + 30, 0, 255)))

            # Smooth slight blur for organic appearance
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
            img_filename = f"{c.lower()}_{i+1:03d}.jpg"
            img.save(class_dir / img_filename, "JPEG", quality=92)

    logger.info(f"Generated {num_samples_per_class * len(classes)} synthetic livestock images across {len(classes)} classes in {output_dir}.")


def generate_synthetic_clinical_data(
    output_file: str = "data/clinical/clinical_data.csv",
    num_samples: int = 350,
    seed: int = 42,
) -> None:
    """Generates realistic clinical and physiological health records."""
    seed_everything(seed)
    np.random.seed(seed)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    breeds = ["Gir", "Sahiwal", "Holstein_Friesian", "Jersey", "Red_Sindhi", "Murrah_Buffalo"]
    sexes = ["Female", "Female", "Female", "Male"]  # realistic herd ratio
    vaccination_status = ["Up_to_date", "Overdue", "Unvaccinated", "Partial"]

    records = []
    diseases = [
        "Lumpy_Skin_Disease",
        "Foot_and_Mouth_Disease",
        "Bovine_Mastitis",
        "Blackleg",
        "Healthy_Cattle",
    ]

    for i in range(num_samples):
        # Pick disease
        disease = np.random.choice(diseases, p=[0.22, 0.22, 0.20, 0.16, 0.20])
        age = round(float(np.random.uniform(1.0, 10.0)), 1)
        breed = np.random.choice(breeds)
        sex = np.random.choice(sexes)
        weight = round(float(np.random.normal(420, 60)), 1)

        # Baseline physiological parameters
        if disease == "Healthy_Cattle":
            temperature = round(float(np.random.normal(38.6, 0.3)), 1)
            heart_rate = int(np.random.normal(65, 6))
            respiratory_rate = int(np.random.normal(24, 4))
            milk_yield_drop = 0.0
            appetite = "Normal"
            activity = "Active"
            vacc_hist = np.random.choice(["Up_to_date", "Partial"], p=[0.8, 0.2])
            symptoms = "none"

        elif disease == "Lumpy_Skin_Disease":
            temperature = round(float(np.random.normal(40.6, 0.5)), 1)  # High fever
            heart_rate = int(np.random.normal(82, 8))
            respiratory_rate = int(np.random.normal(36, 6))
            milk_yield_drop = round(float(np.random.uniform(30.0, 75.0)), 1)
            appetite = np.random.choice(["Reduced", "Anorexia"], p=[0.4, 0.6])
            activity = np.random.choice(["Lethargic", "Depressed"], p=[0.7, 0.3])
            vacc_hist = np.random.choice(vaccination_status, p=[0.1, 0.3, 0.4, 0.2])
            symptoms = "fever,skin_nodules,swollen_lymph_nodes,nasal_discharge,loss_of_appetite"

        elif disease == "Foot_and_Mouth_Disease":
            temperature = round(float(np.random.normal(40.8, 0.6)), 1)  # Severe fever
            heart_rate = int(np.random.normal(88, 10))
            respiratory_rate = int(np.random.normal(38, 7))
            milk_yield_drop = round(float(np.random.uniform(50.0, 90.0)), 1)
            appetite = "Anorexia"
            activity = "Lethargic"
            vacc_hist = np.random.choice(vaccination_status, p=[0.05, 0.35, 0.45, 0.15])
            symptoms = "fever,excessive_salivation,oral_vesicles,lameness,loss_of_appetite"

        elif disease == "Bovine_Mastitis":
            temperature = round(float(np.random.normal(39.8, 0.6)), 1)
            heart_rate = int(np.random.normal(74, 8))
            respiratory_rate = int(np.random.normal(28, 5))
            milk_yield_drop = round(float(np.random.uniform(40.0, 85.0)), 1)
            appetite = np.random.choice(["Normal", "Reduced"], p=[0.3, 0.7])
            activity = "Normal"
            vacc_hist = np.random.choice(vaccination_status, p=[0.4, 0.2, 0.2, 0.2])
            symptoms = "swollen_udder,abnormal_milk_clots,painful_quarter,reduced_milk"

        elif disease == "Blackleg":
            temperature = round(float(np.random.normal(41.2, 0.5)), 1)  # Extreme fever
            heart_rate = int(np.random.normal(96, 12))
            respiratory_rate = int(np.random.normal(42, 8))
            milk_yield_drop = round(float(np.random.uniform(60.0, 100.0)), 1)
            appetite = "Anorexia"
            activity = np.random.choice(["Recumbent", "Depressed"], p=[0.6, 0.4])
            vacc_hist = np.random.choice(vaccination_status, p=[0.05, 0.30, 0.55, 0.10])
            symptoms = "acute_fever,crepitant_swelling,severe_lameness,depression,recumbency"

        # Geospatial & temporal context (for optional cluster analytics)
        # Centered around livestock farming zones (lat ~18.5-22.5 N, lon ~72.8-78.5 E)
        lat = round(float(np.random.uniform(18.5, 21.5)), 4)
        lon = round(float(np.random.uniform(73.5, 77.5)), 4)
        date_str = f"2026-08-{np.random.randint(1, 31):02d}"

        records.append({
            "animal_id": f"LV-{i+1:04d}",
            "age_years": age,
            "breed": breed,
            "sex": sex,
            "weight_kg": weight,
            "body_temperature_c": temperature,
            "heart_rate_bpm": heart_rate,
            "respiratory_rate_bpm": respiratory_rate,
            "milk_yield_drop_pct": milk_yield_drop,
            "appetite": appetite,
            "activity_level": activity,
            "vaccination_status": vacc_hist,
            "symptoms": symptoms,
            "latitude": lat,
            "longitude": lon,
            "recorded_date": date_str,
            "disease": disease,
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    logger.info(f"Generated {len(df)} clinical records in {output_file}.")


def main():
    logger.info("Initializing sample livestock dataset generator...")
    generate_synthetic_livestock_images("data/images", num_samples_per_class=40)
    generate_synthetic_clinical_data("data/clinical/clinical_data.csv", num_samples=350)
    logger.info("Sample datasets successfully created.")


if __name__ == "__main__":
    main()
