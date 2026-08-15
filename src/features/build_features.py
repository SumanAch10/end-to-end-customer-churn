import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
import os

"""
to-do list:
    binary mapping
    one-hot encoding
    
1st priority - binary mapping Done!!

binary columns = unique values == 2(yes/no), male/female
"""
"""
the parameters are validated already, so no need of validation
regarding the length



"""
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RAW = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")
PROCESSED = os.path.join(PROJECT_ROOT, "data/processed/processed.csv")

df = pd.read_csv(RAW)


def build_features(df: pd.DataFrame) -> ColumnTransformer:
    """
    Apply complete feature engineering pipeline for training data.

    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.
    """

    # first task - binary mapping
    # Second task-One hot encoding
    # Third task - Separate numeric columns and add Standard scalar
    cols = df.select_dtypes(include="object").columns.tolist()
    cols.remove("customerID")
    num_cols = df.select_dtypes(include="number").columns
    # separate the numeric and categorical columns

    pre_processor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cols),
            ("numeric", "passthrough", num_cols),
        ]
    )
    return pre_processor


if __name__ == "__main__":
    build_features(df=df, target_col="Churn")
