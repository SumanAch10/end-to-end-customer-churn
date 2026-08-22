"""
Gradio frontend for Telco Customer Churn Prediction.

Responsibilities:
1. Display the user interface.
2. Collect customer information.
3. Send customer data to the FastAPI backend through HTTP.
4. Display the prediction returned by the backend.

This module does NOT import or use the ML model directly.
"""

import os

import gradio as gr
import requests

# ============================================================
# BACKEND CONFIGURATION
# ============================================================

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000/predict")


# ============================================================
# FRONTEND CALLBACK
# ============================================================


def gradio_predict(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):

    # Basic frontend check
    if tenure is None:
        raise gr.Error("Tenure is required.")

    if monthly_charges is None:
        raise gr.Error("Monthly Charges is required.")

    if total_charges is None:
        raise gr.Error("Total Charges is required.")

    # Build request body
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        # Gradio → FastAPI
        response = requests.post(API_URL, json=customer, timeout=10)

    except requests.RequestException as exc:
        raise gr.Error("Could not connect to the prediction backend.") from exc

    # Backend validation error
    if response.status_code == 422:
        raise gr.Error(f"Invalid customer data: {response.text}")

    # Other backend error
    if response.status_code != 200:
        raise gr.Error(
            f"Prediction failed. Backend returned " f"status {response.status_code}."
        )

    result = response.json()

    return (result["label"], result["churn_probability"])


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="Telco Customer Churn Predictor") as demo:

    gr.Markdown("""
        # Telco Customer Churn Predictor

        Enter the customer's information below.

        The application will estimate the probability that the
        customer is likely to churn.
        """)

    with gr.Row():

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column():

            gr.Markdown("### Customer Information")

            gender = gr.Dropdown(
                choices=["Male", "Female"], value="Male", label="Gender"
            )

            senior_citizen = gr.Dropdown(
                choices=[0, 1], value=0, label="Senior Citizen"
            )

            partner = gr.Dropdown(choices=["Yes", "No"], value="No", label="Partner")

            dependents = gr.Dropdown(
                choices=["Yes", "No"], value="No", label="Dependents"
            )

            tenure = gr.Number(value=1, minimum=0, label="Tenure (months)")

            phone_service = gr.Dropdown(
                choices=["Yes", "No"], value="Yes", label="Phone Service"
            )

            multiple_lines = gr.Dropdown(
                choices=["Yes", "No", "No phone service"],
                value="No",
                label="Multiple Lines",
            )

        # ====================================================
        # MIDDLE COLUMN
        # ====================================================

        with gr.Column():

            gr.Markdown("### Internet Services")

            internet_service = gr.Dropdown(
                choices=["DSL", "Fiber optic", "No"],
                value="DSL",
                label="Internet Service",
            )

            online_security = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Online Security",
            )

            online_backup = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Online Backup",
            )

            device_protection = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Device Protection",
            )

            tech_support = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Tech Support",
            )

            streaming_tv = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Streaming TV",
            )

            streaming_movies = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Streaming Movies",
            )

        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column():

            gr.Markdown("### Billing & Contract")

            contract = gr.Dropdown(
                choices=["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract",
            )

            paperless_billing = gr.Dropdown(
                choices=["Yes", "No"], value="Yes", label="Paperless Billing"
            )

            payment_method = gr.Dropdown(
                choices=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                value="Electronic check",
                label="Payment Method",
            )

            monthly_charges = gr.Number(
                value=50.0, minimum=0, label="Monthly Charges ($)"
            )

            total_charges = gr.Number(value=50.0, minimum=0, label="Total Charges ($)")

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = gr.Button("Predict Churn", variant="primary")

    # ========================================================
    # OUTPUT
    # ========================================================

    gr.Markdown("## Prediction Result")

    with gr.Row():

        prediction_output = gr.Textbox(label="Prediction", interactive=False)

        probability_output = gr.Number(label="Churn Probability", interactive=False)

    # ========================================================
    # BUTTON EVENT
    # ========================================================

    predict_button.click(
        fn=gradio_predict,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
        outputs=[prediction_output, probability_output],
    )


# ============================================================
# RUN FRONTEND
# ============================================================

if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)
