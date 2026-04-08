"""Feature engineering for the serving layer.

The derived feature (price_diff) must stay in sync with features/pipeline.py.
The canonical definitions are in features/feature_repo/definitions.py.

Model input: 5 features. Raw prices (actual_price, recommended_price) are
vehicle attributes used to compute price_diff but are NOT model features —
price_diff captures the full pricing signal with less collinearity.
"""

import pandas as pd

FEATURE_COLS = [
    "technology",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived pricing feature."""
    df = df.copy()
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    return df
