"""Master Execution Pipeline for PashuRaksha AI.

Executes all 15 stages in deterministic order:
  Step 1: Discover & setup datasets
  Step 2: Inspect datasets & compute class distributions
  Step 3: Clean images non-destructively
  Step 4: Create stratified train/val/test splits
  Step 5: Train MobileNetV3 image classifier (2-stage transfer learning + calibration)
  Step 6: Evaluate image classifier on test split
  Step 7: Train clinical XGBoost risk model
  Step 8: Extract MobileNetV3 embeddings & train Isolation Forest anomaly detector
  Step 9: Generate visual Grad-CAM and clinical SHAP explanations
  Step 10: Run spatial DBSCAN and temporal outbreak surveillance analytics
  Step 11: Execute multimodal decision-support inference demonstrations
  Step 12: Generate final model evaluation report (reports/final_model_report.md)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger

logger = get_logger("MasterPipeline")


def run_command(cmd: str, desc: str) -> None:
    """Runs a Python script and logs status."""
    logger.info(f">>> [RUNNING] {desc}...")
    logger.info(f"Command: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Failed: {desc}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        raise RuntimeError(f"Step failed: {desc}")
    else:
        if res.stdout.strip():
            print(res.stdout)
        logger.info(f">>> [COMPLETED] {desc}")


def generate_final_model_report() -> None:
    """Synthesizes all inspection, training, calibration, and test metrics into reports/final_model_report.md."""
    logger.info("Generating reports/final_model_report.md...")
    
    # Load inspection report
    insp_data = {}
    if os.path.exists("reports/dataset_report.json"):
        with open("reports/dataset_report.json", "r", encoding="utf-8") as f:
            insp_data = json.load(f)

    # Load image metrics
    img_metrics = {}
    if os.path.exists("reports/image_test_metrics.json"):
        with open("reports/image_test_metrics.json", "r", encoding="utf-8") as f:
            img_metrics = json.load(f)

    # Load clinical metrics
    clin_metrics = {}
    if os.path.exists("reports/clinical_metrics.json"):
        with open("reports/clinical_metrics.json", "r", encoding="utf-8") as f:
            clin_metrics = json.load(f)

    # Load anomaly config
    anomaly_cfg = {}
    if os.path.exists("models/anomaly_model/anomaly_config.json"):
        with open("models/anomaly_model/anomaly_config.json", "r", encoding="utf-8") as f:
            anomaly_cfg = json.load(f)

    img_info = insp_data.get("image_dataset", {})
    clin_info = insp_data.get("clinical_dataset", {})

    report_content = f"""# PashuRaksha AI — Final Model & Pipeline Performance Report

**Date & Time**: 2026-09-15  
**System**: PashuRaksha AI (Livestock Health Screening & Decision Support System)  
**Scope**: Production-Oriented AI/ML Architecture & Evaluation  

---

## 1. Executive Summary & Veterinary Intended Use

PashuRaksha AI is an intelligent, multi-component livestock health screening prototype designed to assist paravets, farmers, and veterinary clinicians in early triage, disease classification, risk estimation, and atypical disease-pattern detection.

> [!IMPORTANT]
> **Clinical Scope Disclaimer**:
> This system is an **AI decision-support screening tool**, not a replacement for a licensed veterinarian. Predictions represent statistical associations and risk estimations rather than confirmed laboratory diagnoses. When confidence is low, symptoms are severe, or unknown patterns are flagged, immediate veterinary clinical examination and diagnostic laboratory confirmation are mandatory.

---

## 2. Dataset Inspection & Integrity Summary

### 2.1 Livestock Image Dataset
- **Directory**: `{img_info.get("dataset_directory", "data/processed/images")}`
- **Total Valid Images**: {img_info.get("total_valid_images", "N/A")}
- **Disease Classes ({img_info.get("num_classes", 0)})**: {", ".join(img_info.get("classes", []))}
- **Class Imbalance Ratio**: {img_info.get("class_imbalance_ratio", "N/A")}
- **Corrupted / Invalid Images Detected**: {img_info.get("corrupted_images_count", 0)}
- **Duplicate Images Filtered**: {img_info.get("duplicate_images_count", 0)}

#### Class Distribution Breakdown:
| Disease Class | Sample Count |
| :--- | :--- |
"""
    for cls, cnt in img_info.get("class_distribution", {}).items():
        report_content += f"| **{cls}** | {cnt} |\n"

    report_content += f"""
### 2.2 Tabular Clinical Dataset
- **File**: `{clin_info.get("file_path", "data/clinical/clinical_data.csv")}`
- **Total Records**: {clin_info.get("total_rows", "N/A")}
- **Total Features**: {clin_info.get("total_columns", "N/A")}
- **Target Diagnosis Column**: `{clin_info.get("target_column", "disease")}`
- **Missing Values**: None (Median/Constant Imputation Applied)
- **Duplicate Records**: {clin_info.get("duplicate_rows", 0)}

---

## 3. Image Classification Model Performance (MobileNetV3-Large)

The primary visual classifier utilizes **MobileNetV3-Large** with ImageNet pretrained weights and 2-stage transfer learning (frozen head training followed by deep feature fine-tuning). Post-hoc **Temperature Scaling** was applied to calibrate probability estimates.

### 3.1 Untouched Test Set Evaluation Metrics:
- **Test Set Size**: {img_metrics.get("test_samples_count", "N/A")} samples
- **Overall Accuracy**: **{img_metrics.get("accuracy", 0.0) * 100:.2f}%**
- **Balanced Accuracy**: **{img_metrics.get("balanced_accuracy", 0.0) * 100:.2f}%**
- **Macro Precision**: {img_metrics.get("macro_precision", 0.0):.4f}
- **Macro Recall**: {img_metrics.get("macro_recall", 0.0):.4f}
- **Macro F1 Score**: **{img_metrics.get("macro_f1", 0.0):.4f}**
- **Weighted F1 Score**: {img_metrics.get("weighted_f1", 0.0):.4f}

### 3.2 Probability Calibration (Temperature Scaling):
- **Learned Temperature Parameter ($T$)**: {img_metrics.get("learned_temperature", 1.0):.4f}
- **Pre-Calibration Expected Calibration Error (ECE)**: {img_metrics.get("raw_ece", 0.0):.4f}
- **Post-Calibration Expected Calibration Error (ECE)**: **{img_metrics.get("calibrated_ece", 0.0):.4f}**

---

## 4. Clinical Symptom Model Performance (XGBoost)

The clinical tabular risk engine utilizes **XGBoost Classifier** trained with sample weights to balance representation across disease categories.

### 4.1 Test Evaluation Metrics:
- **Test Set Size**: {clin_metrics.get("test_samples_count", "N/A")} records
- **Test Accuracy**: **{clin_metrics.get("accuracy", 0.0) * 100:.2f}%**
- **Balanced Accuracy**: **{clin_metrics.get("balanced_accuracy", 0.0) * 100:.2f}%**
- **Macro Precision**: {clin_metrics.get("macro_precision", 0.0):.4f}
- **Macro Recall**: {clin_metrics.get("macro_recall", 0.0):.4f}
- **Macro F1 Score**: **{clin_metrics.get("macro_f1", 0.0):.4f}**

---

## 5. Anomaly & Out-of-Distribution (OOD) Detection

- **Architecture**: MobileNetV3 960-dim latent feature embeddings + **Isolation Forest**
- **Decision Threshold (95th percentile)**: `{anomaly_cfg.get("anomaly_threshold", "N/A")}`
- **Purpose**: Flags atypical disease lesions and unseen pathogens, preventing overconfident false positives on out-of-distribution inputs.

---

## 6. Explainable AI (XAI) Artifacts

1. **Grad-CAM Visual Heatmaps**:
   - Generates gradient-weighted activation overlays on lesion images (`reports/explanations/images/`).
   - Clearly highlights spatial attention while emphasizing that heatmaps reflect statistical network activations rather than confirmed histopathology.
2. **SHAP Feature Importance**:
   - Global summary plots (`reports/explanations/shap_summary.png`) rank physiological features (body temperature, milk yield drop, heart rate, acute symptoms).
   - Local explanations provide feature contribution impacts for individual cases.

---

## 7. Multimodal Fusion & Triage Protocol

The multimodal fusion engine computes a **0–100 Screening Risk Score**:
- **0–30**: Low Risk (Routine monitoring)
- **31–60**: Moderate Risk (Close monitoring / Vet consultation if persistent)
- **61–80**: High Risk (Veterinary consultation & lab confirmation)
- **81–100**: Critical Risk / Safety Override (Emergency veterinary intervention)

---

## 8. Real-World Limitations & Responsible AI Guidelines

1. **Dataset Scale & Diversity**: Models require ongoing validation across diverse breeds, lighting conditions, coat colors, and regional climate variations.
2. **Histopathological Confirmation**: Visual screening cannot differentiate indistinguishable vesicular lesions (e.g. FMD vs Swine Vesicular Disease vs Vesicular Stomatitis) without laboratory PCR/ELISA confirmation.
3. **Safety Override Mandatory**: System strictly enforces non-negotiable veterinary referrals when critical symptoms (hyperpyrexia >= 41.0 C, recumbency, severe respiratory distress) are present.
"""

    os.makedirs("reports", exist_ok=True)
    with open("reports/final_model_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Successfully generated reports/final_model_report.md.")


def main():
    logger.info("================================================================")
    logger.info("   PASHURAKSHA AI — END-TO-END REPRODUCIBLE ML PIPELINE RUNNER  ")
    logger.info("================================================================")

    # Step 1: Check / Setup sample data if empty
    run_command("python data_processing/setup_sample_data.py", "Step 1: Dataset Generation & Initialization")

    # Step 2: Inspect dataset
    run_command("python data_processing/inspect_dataset.py --image-dir data/images --clinical-file data/clinical/clinical_data.csv", "Step 2: Dataset Inspection & Distribution Analysis")

    # Step 3: Clean images
    run_command("python data_processing/clean_images.py --source-dir data/images --dest-dir data/processed/images", "Step 3: Non-Destructive Image Cleaning")

    # Step 4: Stratified Split
    run_command("python data_processing/prepare_images.py --processed-dir data/processed/images --seed 42", "Step 4: Stratified Dataset Partitioning (70/15/15)")

    # Step 5: Train MobileNetV3 Image Model
    run_command("python training/train_image.py --data-dir data/processed/images --epochs-s1 10 --epochs-s2 15 --batch-size 16 --seed 42", "Step 5: MobileNetV3 Image Model Training & Calibration")

    # Step 6: Evaluate Image Model on Test Set
    run_command("python training/evaluate_image.py --model models/image_model/best_model.pth", "Step 6: Image Model Untouched Test Evaluation")

    # Step 7: Train Clinical Model
    run_command("python training/train_clinical.py --data data/clinical/clinical_data.csv --seed 42", "Step 7: Clinical XGBoost Model Training & Evaluation")

    # Step 8: Train Anomaly Detector
    run_command("python training/train_anomaly.py --image-model models/image_model/best_model.pth --seed 42", "Step 8: Latent Embedding Extraction & Isolation Forest Training")

    # Step 9: Generate XAI (Grad-CAM & SHAP)
    # Pick a sample image
    sample_img = "data/processed/images/Lumpy_Skin_Disease/lumpy_skin_disease_001.jpg"
    run_command(f'python explainability/image_explain.py --image "{sample_img}"', "Step 9a: Grad-CAM Image Explainability")
    run_command("python explainability/clinical_explain.py", "Step 9b: SHAP Clinical Feature Explainability")

    # Step 10: Spatial & Temporal Outbreak Analytics
    run_command("python analytics/disease_clusters.py --data data/clinical/clinical_data.csv", "Step 10: Geospatial DBSCAN & Temporal Outbreak Surveillance Analytics")

    # Step 11: Multimodal Inference Demonstrations
    logger.info(">>> Running Multimodal Inference Demonstrations...")
    # Demo 1: Matching High-Risk Case
    cmd_demo1 = (
        'python inference/multimodal_predict.py '
        f'--image "{sample_img}" '
        '--age 4.0 --breed "Gir" --temperature 40.6 --symptoms "fever,skin_nodules,swollen_lymph_nodes,loss_of_appetite"'
    )
    run_command(cmd_demo1, "Step 11a: Multimodal Inference — Matching High-Confidence Case")

    # Demo 2: Emergency Safety Override Case (Severe Recumbency & Hyperpyrexia)
    sample_blackleg = "data/processed/images/Blackleg/blackleg_001.jpg"
    cmd_demo2 = (
        'python inference/multimodal_predict.py '
        f'--image "{sample_blackleg}" '
        '--age 2.5 --breed "Sahiwal" --temperature 41.3 --activity "Recumbent" '
        '--symptoms "acute_fever,crepitant_swelling,severe_lameness,depression,recumbency"'
    )
    run_command(cmd_demo2, "Step 11b: Multimodal Inference — Emergency Safety Override Case")

    # Step 12: Final Model Report Generation
    generate_final_model_report()

    logger.info("================================================================")
    logger.info("   PASHURAKSHA AI PIPELINE FULLY EXECUTED SUCCESSFULLY!         ")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
