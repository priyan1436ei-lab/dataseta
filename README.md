# PashuRaksha AI: Intelligent Livestock Health Screening & Decision Support System

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PashuRaksha AI** is an intelligent, multi-component, reproducible AI/ML pipeline designed for early screening, triage, and risk assessment of livestock diseases in cattle and ruminants.

---

## Intended Use & Veterinary Safety Disclaimer

> [!IMPORTANT]
> **PashuRaksha AI is an AI-powered screening and clinical decision-support prototype, NOT a replacement for a qualified veterinarian.**  
> Neural network predictions represent statistical associations based on visual and physiological observations. They do not constitute definitive medical or laboratory diagnoses. When confidence is low, symptoms are critical, or unknown patterns are detected, immediate veterinary consultation and laboratory diagnostic confirmation (e.g., PCR, ELISA, bacteriological culture) are strongly recommended.

---

## System Architecture

```text
Livestock Image                     Clinical & Symptom Profile
       │                                       │
       ▼                                       ▼
 MobileNetV3-Large                     Clinical Preprocessor
 (2-Stage Transfer)                   (Median Imputation, OHE,
       │                               MultiLabelBinarizer)
       ├──► Calibrated Probabilities           │
       │    (Temperature Scaling)              ▼
       │                                XGBoost Classifier
       ▼                                       │
Feature Embeddings (960-dim)                   ▼
       │                              Clinical Probabilities
       ▼
Isolation Forest OOD Detector
       │
       ▼
Unknown / Unusual Pattern Flag
       │
       └───────────────────┬───────────────────┘
                           │
                           ▼
               Multimodal Fusion Engine
        (Evidence Fusion, Agreement Metric,
         0–100 Screening Risk Score,
         Clinical Emergency Safety Overrides)
                           │
                           ▼
           Structured Screening Output
      ├── Disease Prediction (Top-K Probabilities)
      ├── 0–100 Screening Risk Score (Low / Moderate / High / Critical)
      ├── Unknown Pattern Alert (OOD)
      ├── Model Agreement Score (High / Moderate / Low / Conflicting)
      ├── Key Contributing Factors
      └── Veterinary Referral & Laboratory Recommendation
```

---

## Key Features & Capabilities

1. **Vision Classification (MobileNetV3-Large)**:
   - 2-stage transfer learning (frozen backbone training followed by fine-tuning).
   - Post-hoc **Temperature Scaling** for calibrated probability estimates (ECE minimization).
   - Class-weighted CrossEntropyLoss for handling class imbalances.
   - Early stopping on validation macro-F1.

2. **Clinical Risk Modeling (XGBoost)**:
   - Handles numerical vitals (temperature, heart rate, respiration, milk drop).
   - Handles categorical profiles (breed, sex, vaccination status).
   - Multi-value symptom encoding (`MultiLabelBinarizer`).
   - Stratified, leakage-proof train/test evaluation.

3. **Anomaly & Out-of-Distribution Detection**:
   - Extracts 960-dimensional latent embeddings from MobileNetV3.
   - Fits an **Isolation Forest** on normal feature distributions to detect rare/unseen pathogens or anomalous lesions.

4. **Transparent Multimodal Fusion & Decision Support**:
   - Weighted evidential fusion of image + clinical probabilities.
   - Calculates cross-modality agreement to highlight discrepancies.
   - Computes a calibrated **0–100 Screening Risk Score** and categorical level (`Low`, `Moderate`, `High`, `Critical`).
   - **Safety Overrides**: Automatically forces critical emergency warnings on high-acuity vitals (e.g., hyperpyrexia $\ge 41.0^\circ\text{C}$, recumbency, severe respiratory distress).

5. **Explainable AI (XAI)**:
   - **Grad-CAM**: Generates class activation heatmap overlays for MobileNetV3.
   - **SHAP**: Generates global and local feature importance rankings for XGBoost.

6. **Surveillance & Outbreak Analytics**:
   - Geospatial clustering via **DBSCAN (Haversine metric)**.
   - Rolling temporal trend monitoring across geographic regions.

---

## Directory Structure

```text
pashuraksha-ai/
├── data/
│   ├── raw/
│   ├── images/
│   ├── clinical/
│   └── processed/
│       └── images/
│
├── data_processing/
│   ├── inspect_dataset.py       # Dataset discovery, statistics & distribution plotting
│   ├── clean_images.py          # Non-destructive corrupted & duplicate image cleaning
│   ├── prepare_images.py        # Stratified 70/15/15 dataset splitting
│   ├── prepare_clinical.py      # Tabular preprocessing pipeline & transformer persistence
│   └── setup_sample_data.py     # Representative livestock dataset generator
│
├── training/
│   ├── train_image.py           # 2-stage MobileNetV3 training & calibration
│   ├── evaluate_image.py        # Test split evaluation, confusion matrix & metrics
│   ├── train_clinical.py        # XGBoost clinical tabular risk modeling
│   ├── train_anomaly.py         # Latent embedding extraction & Isolation Forest fitting
│   └── evaluate_clinical.py
│
├── inference/
│   ├── image_predict.py         # Standalone calibrated image inference CLI
│   ├── clinical_predict.py      # Standalone clinical symptom inference CLI
│   └── multimodal_predict.py    # End-to-end multimodal decision-support CLI
│
├── explainability/
│   ├── image_explain.py         # Grad-CAM heatmap generator
│   └── clinical_explain.py      # SHAP feature importance & case explainer
│
├── analytics/
│   └── disease_clusters.py      # Geospatial DBSCAN & temporal surveillance
│
├── models/
│   ├── image_model/             # Saved best_model.pth checkpoint
│   ├── clinical_model/          # Saved XGBoost model.joblib & preprocessor.joblib
│   └── anomaly_model/           # Saved image_anomaly.joblib & anomaly_config.json
│
├── reports/
│   ├── image_training/          # Loss, accuracy, F1, learning rate curves
│   ├── explanations/            # Grad-CAM heatmaps and SHAP summary plots
│   ├── dataset_report.json
│   ├── dataset_split.csv
│   ├── image_test_metrics.json
│   ├── clinical_metrics.json
│   └── final_model_report.md    # Synthesized final evaluation report
│
├── configs/
│   └── default_config.yaml
│
├── utils/
│   ├── seed.py                  # Deterministic random seeding
│   ├── metrics.py               # Classification & calibration metrics
│   └── logger.py                # Structured console & file logging
│
├── run_pipeline.py              # Master end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Create Virtual Environment

**Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## End-to-End Execution Workflow

You can execute the entire pipeline with a single command:

```bash
python run_pipeline.py
```

Or run each modular step individually:

### Step 1: Inspect Dataset
```bash
python data_processing/inspect_dataset.py \
  --image-dir data/images \
  --clinical-file data/clinical/clinical_data.csv
```

### Step 2: Clean Dataset (Non-Destructive)
```bash
python data_processing/clean_images.py \
  --source-dir data/images \
  --dest-dir data/processed/images
```

### Step 3: Stratified 70/15/15 Splitting
```bash
python data_processing/prepare_images.py \
  --processed-dir data/processed/images \
  --seed 42
```

### Step 4: Train MobileNetV3 Image Model
```bash
python training/train_image.py \
  --data-dir data/processed/images \
  --epochs-s1 10 \
  --epochs-s2 15 \
  --batch-size 16
```

### Step 5: Evaluate Image Model on Test Split
```bash
python training/evaluate_image.py \
  --model models/image_model/best_model.pth
```

### Step 6: Train Clinical XGBoost Model
```bash
python training/train_clinical.py \
  --data data/clinical/clinical_data.csv
```

### Step 7: Train Anomaly / OOD Detector
```bash
python training/train_anomaly.py \
  --image-model models/image_model/best_model.pth
```

### Step 8: Visual (Grad-CAM) & Clinical (SHAP) Explainability
```bash
python explainability/image_explain.py \
  --image data/processed/images/Lumpy_Skin_Disease/lumpy_skin_disease_001.jpg

python explainability/clinical_explain.py
```

### Step 9: Geospatial Outbreak & Cluster Surveillance
```bash
python analytics/disease_clusters.py \
  --data data/clinical/clinical_data.csv
```

---

## CLI Inference Demonstrations

### 1. Standalone Image Inference
```bash
python inference/image_predict.py \
  --image data/processed/images/Lumpy_Skin_Disease/lumpy_skin_disease_001.jpg
```

### 2. Standalone Clinical Inference
```bash
python inference/clinical_predict.py \
  --temperature 40.6 \
  --symptoms "fever,skin_nodules,loss_of_appetite" \
  --breed "Gir" \
  --age 4.0
```

### 3. Full Multimodal Decision-Support Inference
```bash
python inference/multimodal_predict.py \
  --image data/processed/images/Lumpy_Skin_Disease/lumpy_skin_disease_001.jpg \
  --age 4.0 \
  --breed "Gir" \
  --temperature 40.6 \
  --symptoms "fever,skin_nodules,swollen_lymph_nodes,loss_of_appetite"
```

#### Example Output:
```json
{
  "screening_prediction": "Possible Lumpy_Skin_Disease",
  "top_disease_probabilities": {
    "Lumpy_Skin_Disease": 0.8845,
    "Foot_and_Mouth_Disease": 0.0512,
    "Bovine_Mastitis": 0.0321
  },
  "image_confidence": 0.8920,
  "clinical_probability": 0.8770,
  "unknown_pattern": false,
  "anomaly_score": 0.1824,
  "risk_score": 76,
  "risk_level": "High",
  "model_agreement": "High",
  "safety_overrides_triggered": false,
  "contributing_factors": [
    "Image visual pattern associated with Lumpy_Skin_Disease (89.2% confidence)",
    "Elevated body temperature (40.6°C)",
    "Reported symptoms: fever, skin nodules, swollen lymph nodes, loss of appetite"
  ],
  "recommended_action": "Veterinary consultation recommended",
  "vet_referral": true,
  "lab_confirmation": "Recommended",
  "disclaimer": "This is an AI screening and decision-support result and not a confirmed veterinary diagnosis."
}
```

---

## Reproducibility & Global Seeds

All random number generators (Python `random`, `numpy`, `torch`, `torch.cuda`, cuDNN deterministic flags, scikit-learn, XGBoost) are configured with `seed=42`. All dataset splitting, model training, and embedding extraction procedures are strictly deterministic.

---

## License

This project is developed under the MIT License.
# dataseta
"# dataset" 
