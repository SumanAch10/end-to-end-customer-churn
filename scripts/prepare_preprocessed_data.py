import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


RAW = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")
OUT = os.path.join(PROJECT_ROOT, "data/processed/telco_churn_processed.csv")

# 1)load dataset and convert into a dataframe
df = pd.read_csv(RAW)

status, failed = validate_telco_data(df=df)

if not status:
    raise ValueError("Data validation failed. Training stopped")

# 2) preprocess (drops id, fixes totalcharges, etc.)
df = preprocess_data(df=df, target_col="Churn")

# 3) Ensure the target is 0/1 only if still object
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

# Sanity Check
assert df["Churn"].isna().sum() == 0, "Churn has NaN after preprocess data"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

# 5) Create dir to save the processed output
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 6) Save the dataframe to the csv file
df.to_csv(OUT, index=False)

print(f"Processed dataset saved to {OUT} | Shape{df.shape}")
