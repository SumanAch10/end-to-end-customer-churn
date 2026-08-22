"""
FastAPI backend for Telco Customer Churn prediction.

Responsibilities:
1. Expose HTTP endpoints.
2. Validate incoming customer data using Pydantic.
3. Pass validated customer data to the inference layer.
4. Return prediction results as JSON.
5. Handle API-level errors and logging.

This module contains NO frontend logic and NO ML training logic.

Architecture:

Client / Gradio
      ↓
POST /predict
      ↓
FastAPI
      ↓
Pydantic validation
      ↓
inference.predict()
      ↓
fitted sklearn Pipeline
      ↓
prediction response
"""

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.inference import predict
from src.utils.utils import setup_logger

# ============================================================
# LOGGING
# ============================================================

logger = setup_logger(
    name=__name__,
    log_file="logs/api.log",
    level=logging.INFO,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Telco Customer Churn API",
    description="Backend API for predicting customer churn.",
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================


class CustomerData(BaseModel):
    """
    Raw customer features expected by the fitted ML pipeline.

    FastAPI + Pydantic validate the request before the data
    reaches the inference layer.
    """

    gender: Literal["Male", "Female"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, description="Number of months the customer has stayed.")

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal["Yes", "No", "No phone service"]

    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: Literal["Yes", "No", "No internet service"]

    OnlineBackup: Literal["Yes", "No", "No internet service"]

    DeviceProtection: Literal["Yes", "No", "No internet service"]

    TechSupport: Literal["Yes", "No", "No internet service"]

    StreamingTV: Literal["Yes", "No", "No internet service"]

    StreamingMovies: Literal["Yes", "No", "No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0)

    TotalCharges: float = Field(ge=0)


# ============================================================
# RESPONSE SCHEMA
# ============================================================


class PredictionResponse(BaseModel):
    """
    Structure returned by the prediction API.
    """

    prediction: int
    label: str
    churn_probability: float
    threshold: float


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/")
def health_check():
    """
    Used to verify that the backend service is running.
    """

    return {"status": "ok", "service": "telco-churn-api"}


# ============================================================
# PREDICTION ENDPOINT
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(customer: CustomerData):
    """
    Receive customer data through HTTP, validate it,
    call the inference layer, and return the prediction.
    """

    try:
        logger.info("Prediction request received.")

        # Pydantic model -> normal Python dictionary
        customer_dict = customer.model_dump()

        # ML logic belongs entirely in inference.py
        result = predict(customer_dict)

        logger.info(
            "Prediction completed successfully | " "prediction=%s | probability=%.4f",
            result["prediction"],
            result["churn_probability"],
        )

        return result

    except ValueError as exc:

        logger.warning("Invalid customer data received: %s", exc)

        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:

        logger.exception("Prediction request failed.")

        raise HTTPException(
            status_code=500, detail="Prediction could not be completed."
        ) from exc
