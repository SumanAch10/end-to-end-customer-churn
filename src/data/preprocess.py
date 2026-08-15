import pandas as pd

"""
list of things to be preprocessed

- total charges to float
- 
"""


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """

    # Remove the whitespaces
    df.columns = df.columns.str.strip()

    check_col = df.columns.tolist()

    # drop ids if present
    for col in check_col:
        if "customer" in col:
            df = df.drop(columns=[col])

    # Convert total charges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert the target into 0/1
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    nums_cols = df.select_dtypes(include=["number"]).columns
    df[nums_cols] = df[nums_cols].fillna(0)

    return df
