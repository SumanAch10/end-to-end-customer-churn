from src.serving.inference import predict

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


def test_predict_returns_valid_result():

    result = predict(VALID_CUSTOMER)

    assert "prediction" in result
    assert "label" in result
    assert "churn_probability" in result
    assert "threshold" in result

    assert result["prediction"] in [0, 1]

    assert 0.0 <= result["churn_probability"] <= 1.0

    assert result["threshold"] == 0.30
