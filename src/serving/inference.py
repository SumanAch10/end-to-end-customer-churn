"""
Inference module for Telco Customer Churn.

Responsibilities:

1. Load the fitted sklearn Pipeline once at application startup.
2. Receive raw customer features.
3. Verify that the expected model input columns are present.
4. Run preprocessing + RandomForest inference through the fitted Pipeline.
5. Apply the configured classification threshold.
6. Return prediction and churn probability.

Training is NEVER performed in this module.

Deployment behavior:

- Locally:
    loads artifacts/production_model

- Inside Docker:
    loads /app/model

The model artifact is stored in MLflow format, but inference does
NOT require an MLflow tracking server.
"""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.utils import setup_logger

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "production_model"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "inference.log"


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(DEFAULT_MODEL_PATH),
)

CHURN_THRESHOLD = float(
    os.getenv(
        "CHURN_THRESHOLD",
        "0.30",
    )
)


# ============================================================
# LOGGING
# ============================================================

logger = setup_logger(
    name=__name__,
    log_file=str(LOG_FILE),
    level=logging.INFO,
)


# ============================================================
# MODEL LOADING
# ============================================================


def _load_pipeline() -> Pipeline:
    """
    Load the fitted production sklearn Pipeline from local storage.

    The model is loaded once when this module is imported.
    The same fitted pipeline is then reused for every prediction.
    """

    try:
        logger.info(
            "Loading production model from: %s",
            MODEL_PATH,
        )

        fitted_pipeline = mlflow.sklearn.load_model(MODEL_PATH)

        if not isinstance(fitted_pipeline, Pipeline):
            raise TypeError(
                f"Expected sklearn Pipeline, "
                f"but loaded {type(fitted_pipeline).__name__}"
            )

        logger.info("Production pipeline loaded successfully.")

        return fitted_pipeline

    except Exception as exc:
        logger.exception("Failed to load production pipeline.")

        raise RuntimeError(f"Unable to load model from {MODEL_PATH}") from exc


# Load ONCE when application starts/imports this module
pipeline = _load_pipeline()


# ============================================================
# MODEL INPUT SCHEMA
# ============================================================


def _get_expected_columns() -> list[str]:
    """
    Read the raw feature columns expected by the fitted preprocessor.
    """

    try:
        preprocessor = pipeline.named_steps["preprocessor"]

        return list(preprocessor.feature_names_in_)

    except (KeyError, AttributeError) as exc:
        raise RuntimeError("Unable to determine expected model input columns.") from exc


EXPECTED_COLUMNS = _get_expected_columns()


# ============================================================
# INPUT PREPARATION
# ============================================================


def _prepare_customer(customer_data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw customer input into the one-row DataFrame
    expected by the fitted sklearn Pipeline.

    No one-hot encoding or model preprocessing occurs here.
    Those transformations are already contained inside the
    fitted Pipeline.
    """

    if not isinstance(customer_data, dict):
        raise TypeError("customer_data must be a dictionary.")

    customer_df = pd.DataFrame([customer_data])

    missing_columns = set(EXPECTED_COLUMNS) - set(customer_df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required model features: " + ", ".join(sorted(missing_columns))
        )

    # Keep only model features and preserve training column order.
    customer_df = customer_df[EXPECTED_COLUMNS]

    return customer_df


# ============================================================
# INFERENCE
# ============================================================


def predict(customer_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict churn for one customer.

    Returns:
        prediction:
            0 or 1

        label:
            Human-readable prediction

        churn_probability:
            Probability of churn/class 1

        threshold:
            Classification threshold used
    """

    try:
        customer_df = _prepare_customer(customer_data)

        # Pipeline automatically performs:
        #
        # raw customer
        #       ↓
        # fitted ColumnTransformer
        #       ↓
        # fitted RandomForest
        #       ↓
        # class probabilities

        probabilities = pipeline.predict_proba(customer_df)

        churn_probability = float(probabilities[0, 1])

        prediction = int(churn_probability >= CHURN_THRESHOLD)

        label = "Likely to churn" if prediction == 1 else "Not likely to churn"

        logger.info(
            "Prediction completed | "
            "prediction=%s | "
            "probability=%.4f | "
            "threshold=%.2f",
            prediction,
            churn_probability,
            CHURN_THRESHOLD,
        )

        return {
            "prediction": prediction,
            "label": label,
            "churn_probability": churn_probability,
            "threshold": CHURN_THRESHOLD,
        }

    except Exception:
        logger.exception("Customer churn inference failed.")
        raise
