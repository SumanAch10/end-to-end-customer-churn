from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0,
}


def test_health_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_predict_endpoint():

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result


def test_invalid_customer_rejected():

    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["gender"] = "Invalid"

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
