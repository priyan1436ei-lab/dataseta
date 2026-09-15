"""Clinical Tabular Data Preprocessing Pipeline for PashuRaksha AI.

Processes structured clinical & symptom records with:
- Median imputation and standard scaling for numerical features (e.g. age, temperature, weight).
- Categorical one-hot encoding for breeds, sex, vaccination history.
- MultiLabelBinarizer / multi-hot encoding for multi-value symptom strings.
- Leakage-proof fitting strictly on the training partition.
- Exports fitted preprocessor to models/clinical_model/preprocessor.joblib.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
from utils.seed import seed_everything

logger = get_logger("PrepareClinical")


class ClinicalPreprocessor:
    """End-to-end preprocessor for livestock clinical health records."""

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.mlb = MultiLabelBinarizer()
        self.target_encoder = LabelEncoder()

        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.symptom_col: str = ""
        self.target_col: str = ""
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _parse_symptoms(self, series: pd.Series) -> List[List[str]]:
        """Parses comma-separated or list-formatted symptom values."""
        parsed = []
        for val in series:
            if pd.isna(val) or val == "" or str(val).lower() == "none":
                parsed.append([])
            elif isinstance(val, list):
                parsed.append([str(s).strip().lower() for s in val])
            else:
                items = [s.strip().lower() for s in str(val).split(",") if s.strip()]
                parsed.append(items)
        return parsed

    def fit(self, df: pd.DataFrame, target_col: str, symptom_col: str = "symptoms") -> "ClinicalPreprocessor":
        """Fits preprocessors strictly on the training dataframe."""
        self.target_col = target_col
        self.symptom_col = symptom_col

        # Exclude non-clinical metadata/geospatial surveillance columns from feature predictors
        metadata_cols = {"animal_id", "id", "latitude", "longitude", "recorded_date", "date", "lat", "lon", target_col}
        feature_cols = [c for c in df.columns if c.lower() not in metadata_cols]
        feature_df = df[feature_cols].copy()

        # Identify column types
        self.numerical_cols = list(feature_df.select_dtypes(include=[np.number]).columns)
        all_obj_cols = list(feature_df.select_dtypes(include=["object", "category", "string"]).columns)

        if symptom_col in all_obj_cols:
            self.categorical_cols = [c for c in all_obj_cols if c != symptom_col]
        else:
            self.categorical_cols = all_obj_cols
            self.symptom_col = ""

        # Fit Numerical
        if self.numerical_cols:
            num_data = feature_df[self.numerical_cols].values
            num_imputed = self.num_imputer.fit_transform(num_data)
            self.num_scaler.fit(num_imputed)

        # Fit Categorical
        if self.categorical_cols:
            cat_data = feature_df[self.categorical_cols].astype(str).values
            cat_imputed = self.cat_imputer.fit_transform(cat_data)
            self.cat_encoder.fit(cat_imputed)

        # Fit Symptoms Multi-hot
        if self.symptom_col and self.symptom_col in df.columns:
            symptoms_list = self._parse_symptoms(df[self.symptom_col])
            self.mlb.fit(symptoms_list)

        # Fit Target
        if target_col in df.columns:
            self.target_encoder.fit(df[target_col].astype(str))

        # Build feature names
        names = []
        names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_feature_names = self.cat_encoder.get_feature_names_out(self.categorical_cols)
            names.extend(list(cat_feature_names))
        if self.symptom_col and hasattr(self.mlb, "classes_"):
            symptom_names = [f"symptom_{c}" for c in self.mlb.classes_]
            names.extend(symptom_names)
        self.feature_names = names

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms features and optional target."""
        if not self.is_fitted:
            raise ValueError("ClinicalPreprocessor must be fitted before transform.")

        df = df.copy()
        # Ensure all expected columns exist (fill missing with NaN for imputer)
        for col in self.numerical_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = "Unknown"

        parts = []

        # Transform Numerical
        if self.numerical_cols:
            num_data = df[self.numerical_cols].values
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.num_scaler.transform(num_imputed)
            parts.append(num_scaled)

        # Transform Categorical
        if self.categorical_cols:
            cat_data = df[self.categorical_cols].astype(str).values
            cat_imputed = self.cat_imputer.transform(cat_data)
            cat_encoded = self.cat_encoder.transform(cat_imputed)
            parts.append(cat_encoded)

        # Transform Symptoms
        if self.symptom_col and self.symptom_col in df.columns:
            symptoms_list = self._parse_symptoms(df[self.symptom_col])
            symptom_encoded = self.mlb.transform(symptoms_list)
            parts.append(symptom_encoded)

        X = np.hstack(parts) if parts else np.empty((len(df), 0))

        # Transform Target if available
        y = None
        if self.target_col and self.target_col in df.columns:
            y = self.target_encoder.transform(df[self.target_col].astype(str))

        return X, y

    def transform_single_record(self, record_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single dictionary input for inference."""
        df = pd.DataFrame([record_dict])
        X, _ = self.transform(df)
        return X


def process_clinical_dataset(
    file_path: str = "data/clinical/clinical_data.csv",
    output_dir: str = "models/clinical_model",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ClinicalPreprocessor]:
    """Loads, splits, and preprocesses the clinical dataset."""
    seed_everything(seed)
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Clinical file not found: {file_path}")

    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
    logger.info(f"Loaded clinical data with {len(df)} rows, {len(df.columns)} columns.")

    # Identify target column
    target_col = None
    candidate_targets = ["disease", "target", "diagnosis", "condition", "risk_level", "health_status", "label"]
    for col in df.columns:
        if col.lower() in candidate_targets:
            target_col = col
            break

    if not target_col:
        raise ValueError(f"Could not identify target disease column among: {list(df.columns)}")

    symptom_col = "symptoms" if "symptoms" in df.columns else ""

    # Stratified split 70% train / 15% val / 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[target_col])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df[target_col])

    # Fit preprocessor strictly on train_df
    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(train_df, target_col=target_col, symptom_col=symptom_col)

    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    # Save preprocessor
    os.makedirs(output_dir, exist_ok=True)
    prep_path = Path(output_dir) / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path)
    logger.info(f"Saved fitted clinical preprocessor to: {prep_path}")
    logger.info(f"Processed Feature dimension: {X_train.shape[1]} features")

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Clinical Data Preparation")
    parser.add_argument("--data", type=str, default="data/clinical/clinical_data.csv", help="Path to clinical CSV")
    parser.add_argument("--output-dir", type=str, default="models/clinical_model", help="Directory to save preprocessor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    process_clinical_dataset(args.data, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
