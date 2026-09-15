# PashuRaksha AI — Final Model & Pipeline Performance Report

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
- **Directory**: `C:\Users\priya\Downloads\dataset\data\images`
- **Total Valid Images**: 200
- **Disease Classes (5)**: Blackleg, Bovine_Mastitis, Foot_and_Mouth_Disease, Healthy_Cattle, Lumpy_Skin_Disease
- **Class Imbalance Ratio**: 1.0
- **Corrupted / Invalid Images Detected**: 0
- **Duplicate Images Filtered**: 0

#### Class Distribution Breakdown:
| Disease Class | Sample Count |
| :--- | :--- |
| **Blackleg** | 40 |
| **Bovine_Mastitis** | 40 |
| **Foot_and_Mouth_Disease** | 40 |
| **Healthy_Cattle** | 40 |
| **Lumpy_Skin_Disease** | 40 |

### 2.2 Tabular Clinical Dataset
- **File**: `C:\Users\priya\Downloads\dataset\data\clinical\clinical_data.csv`
- **Total Records**: 350
- **Total Features**: 17
- **Target Diagnosis Column**: `disease`
- **Missing Values**: None (Median/Constant Imputation Applied)
- **Duplicate Records**: 0

---

## 3. Image Classification Model Performance (MobileNetV3-Large)

The primary visual classifier utilizes **MobileNetV3-Large** with ImageNet pretrained weights and 2-stage transfer learning (frozen head training followed by deep feature fine-tuning). Post-hoc **Temperature Scaling** was applied to calibrate probability estimates.

### 3.1 Untouched Test Set Evaluation Metrics:
- **Test Set Size**: 30 samples
- **Overall Accuracy**: **100.00%**
- **Balanced Accuracy**: **100.00%**
- **Macro Precision**: 1.0000
- **Macro Recall**: 1.0000
- **Macro F1 Score**: **1.0000**
- **Weighted F1 Score**: 1.0000

### 3.2 Probability Calibration (Temperature Scaling):
- **Learned Temperature Parameter ($T$)**: 1.2615
- **Pre-Calibration Expected Calibration Error (ECE)**: 0.0008
- **Post-Calibration Expected Calibration Error (ECE)**: **0.0037**

---

## 4. Clinical Symptom Model Performance (XGBoost)

The clinical tabular risk engine utilizes **XGBoost Classifier** trained with sample weights to balance representation across disease categories.

### 4.1 Test Evaluation Metrics:
- **Test Set Size**: 53 records
- **Test Accuracy**: **100.00%**
- **Balanced Accuracy**: **100.00%**
- **Macro Precision**: 1.0000
- **Macro Recall**: 1.0000
- **Macro F1 Score**: **1.0000**

---

## 5. Anomaly & Out-of-Distribution (OOD) Detection

- **Architecture**: MobileNetV3 960-dim latent feature embeddings + **Isolation Forest**
- **Decision Threshold (95th percentile)**: `0.45216714268364044`
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
