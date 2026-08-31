"""
Cloud Run entrypoint that serves the API and Gradio UI from one container.
"""

import logging
from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.app.gradio_ui import build_demo
from src.serving.inference import predict
from src.utils.utils import setup_logger

logger = setup_logger(
    name=__name__,
    log_file="logs/cloud-api.log",
    level=logging.INFO,
)


class CustomerData(BaseModel):
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


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float
    threshold: float


app = FastAPI(
    title="Telco Customer Churn",
    description="Combined API and Gradio UI for Cloud Run.",
    version="1.0.0",
)


# NOTE: not "/healthz" -- Cloud Run's Google Frontend intercepts that exact
# path and returns its own 404, so the request never reaches this container.
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "telco-churn-cloudrun"}


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(customer: CustomerData):
    try:
        result = predict(customer.model_dump())
        logger.info(
            "Prediction completed successfully | prediction=%s | probability=%.4f",
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


demo = build_demo(predict_fn=predict)
app = gr.mount_gradio_app(app, demo, path="/")
