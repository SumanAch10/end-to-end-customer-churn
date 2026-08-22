"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data  # Data loading with error handling
from src.data.preprocess import preprocess_data  # Basic data cleaning
from src.features.build_features import build_features

# Feature engineering (CRITICAL for model performance)
from src.utils.validate_telco_data import validate_telco_data  # Data quality validation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")
PROCESSED = os.path.join(PROJECT_ROOT, "data/processed/telco_churn_processed.csv")

sys.path.append(PROJECT_ROOT)


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    """

    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)

    mlflow_db = os.path.join(PROJECT_ROOT, "mlflow.db")
    # Local file-based tracking

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    run = client.get_run("9e45df2952c54d40b2ce67f7e32ed16e")
    params = run.data.params

    # Set the experiment name
    mlflow.set_experiment("Telco Churn-Production Model Training(Tuned Parameters)")
    with mlflow.start_run():
        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(RAW)  # Load raw CSV data with error handling
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            # Log validation failures for debugging
            import json

            mlflow.log_text(
                json.dumps(failed, indent=2), artifact_file="failed_expectations.json"
            )
            raise ValueError(f" Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print(" Preprocessing data...")
        df_before = len(df)
        df = preprocess_data(
            df
        )  # Basic cleaning (handle missing values, fix data types)
        df_after = len(df)

        # Sanity check(Make sure no rows are dropped)
        assert (
            df_after == df_before
        ), "Preprocessing unexpectedly change the number of rows"

        # Save processed dataset for reproducibility and debugging

        os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
        df.to_csv(PROCESSED, index=False)
        print(f"Processed dataset saved to {PROCESSED} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        # == STAGE:4 Split the dataset + Feature engineering
        # 1.Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # 2.Train-test split
        # Second split: divide the 40% equally
        # 20% validation + 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=int(params["random_state"]),
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=int(params["random_state"]),
        )

        # 3.Build preprocessing configuration using training feature
        preprocessor = build_features(X_train)
        rf_model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
            n_jobs=int(params["n_jobs"]),
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf_model)])
        # -------------------------

        # 1. LOG PARAMETERS
        # -------------------------

        mlflow.log_params(
            {
                # Model
                "model": "RandomForestClassifier",
                # Tuned hyperparameters
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"],
                "max_features": params["max_features"],
                # Fixed model configuration
                "class_weight": params["class_weight"],
                "random_state": params["random_state"],
                "n_jobs": params["n_jobs"],
                # Prediction configuration
                "threshold": args.threshold,
                # Data split configuration
                "test_size": args.test_size,
            }
        )

        # -------------------------
        # 2. TRAIN PIPELINE
        # -------------------------

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        mlflow.log_metric("train_time_seconds", train_time)

        # -------------------------
        # 3. MAKE PREDICTIONS
        # -------------------------

        # Probability of churn = class 1
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        y_pred_val = (y_proba_val >= args.threshold).astype(int)

        # -------------------------------------------
        # 4. CALCULATE METRICS FOR THE VALIDATION SET
        # ------------------------------------------

        precision_val = precision_score(y_val, y_pred_val)
        recall_val = recall_score(y_val, y_pred_val)
        f1_val = f1_score(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_proba_val)

        # -------------------------
        # 5. LOG METRICS
        # -------------------------

        mlflow.log_metric("precision_val", precision_val)
        mlflow.log_metric("recall_val", recall_val)
        mlflow.log_metric("f1_val", f1_val)
        mlflow.log_metric("roc_auc_val", auc_val)

        # -------------------------------------------
        # 4. CALCULATE METRICS FOR THE TESTING SET
        # ------------------------------------------
        # Probability of churn = class 1
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= args.threshold).astype(int)

        mlflow.log_metric("Threshold", args.threshold)

        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)

        # -------------------------
        # 5. LOG METRICS
        # -------------------------

        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("recall_test", recall_test)
        mlflow.log_metric("f1_test", f1_test)
        mlflow.log_metric("roc_auc_test", roc_auc_test)

        auprc_val = average_precision_score(y_val, y_proba_val)
        auprc_test = average_precision_score(y_test, y_proba_test)

        mlflow.log_metric("auprc_val", auprc_val)
        mlflow.log_metric("auprc_test", auprc_test)

        # -------------------------
        # 6. LOG FITTED PIPELINE
        # -------------------------

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run churn pipeline with RandomForest + MLflow"
    )
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--test_size", type=float, default=0.4)
    args = p.parse_args()
    main(args)
