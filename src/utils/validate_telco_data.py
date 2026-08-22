import great_expectations as gx
from typing import Tuple, List
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RAW = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")

df = pd.read_csv(RAW)

# context = gx.get_context()

# data_source = context.data_sources.add_pandas("pandas")
# data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")


# batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
# batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """
    print("🔍 Starting data validation with Great Expectations...")

    # 1. Create / get the GX context
    context = gx.get_context()

    # 2. Create a Pandas Data Source
    data_source = context.data_sources.add_pandas(name="telco_pandas_source")

    # 3. Create a Data Asset
    data_asset = data_source.add_dataframe_asset(name="telco_churn_asset")

    # 4. Create a Batch Definition
    # For a Pandas DataFrame, use the whole DataFrame as one batch
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="whole_telco_dataframe"
    )

    # 5. Pass the actual DataFrame at runtime
    batch_parameters = {"dataframe": df}

    # 6. Get the actual Batch
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="tenure", min_value=0, max_value=120
    )

    suite = gx.ExpectationSuite(name="telco_data_suite")

    required_non_null_cols = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "PaymentMethod",
        "Churn",
    ]

    # ============================================================
    # CATEGORICAL DOMAIN RULES
    # ============================================================

    # Yes / No columns
    yes_no_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]

    for col in yes_no_cols:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=["Yes", "No"]
            )
        )

    # MultipleLines
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="MultipleLines", value_set=["Yes", "No", "No phone service"]
        )
    )

    # InternetService
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="InternetService", value_set=["DSL", "Fiber optic", "No"]
        )
    )

    # Internet-related service columns
    internet_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in internet_service_cols:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=["Yes", "No", "No internet service"]
            )
        )

    # Contract
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Contract", value_set=["Month-to-month", "One year", "Two year"]
        )
    )

    # PaymentMethod
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PaymentMethod",
            value_set=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    )

    # Churn target
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Churn", value_set=["Yes", "No"]
        )
    )

    # ============================================================
    # NUMERIC / BUSINESS RANGE RULES
    # ============================================================

    # tenure should be between 0 and 120 months
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure", min_value=0, max_value=120
        )
    )

    # MonthlyCharges should be non-negative
    # and within the reasonable range used by this project
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="MonthlyCharges", min_value=0, max_value=200
        )
    )

    # ============================================================
    # TotalCharges RAW-DATA RULE
    # ============================================================

    # TotalCharges exists, but raw data can contain blank strings.
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="TotalCharges"))

    # Allow either:
    #   - blank / whitespace
    #   - a non-negative integer or decimal number stored as text
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="TotalCharges", regex=r"^\s*$|^\s*\d+(\.\d+)?\s*$"
        )
    )

    # ============================================================
    # VALIDATE COMPLETE SUITE
    # ============================================================

    result = batch.validate(suite)
    print(result.results)
    failed_expectations = [
        str[r.expectation_config] for r in result.results if not r.success
    ]

    return result.success, failed_expectations


if __name__ == "__main__":
    print(validate_telco_data(df=df))
