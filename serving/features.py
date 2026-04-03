"""Feature engineering — must stay in sync with training/train.py."""

import pandas as pd

FEATURE_COLS = [
    "technology",
    "actual_price",
    "recommended_price",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
    "price_ratio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived pricing features."""
    df = df.copy()
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"]
    return df
