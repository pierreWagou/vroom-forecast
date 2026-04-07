"""Tests for the training pipeline — pure logic, no MLflow needed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from train import (
    FEATURE_COLS,
    TARGET_COL,
    load_from_feature_store,
    train_and_evaluate,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal DataFrame with all features + target."""
    df = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "technology": [1, 0, 1, 0],
            "actual_price": [45.0, 30.0, 60.0, 25.0],
            "recommended_price": [50.0, 25.0, 55.0, 30.0],
            "num_images": [8, 2, 15, 0],
            "street_parked": [0, 1, 0, 1],
            "description": [250, 50, 400, 10],
            "num_reservations": [3, 1, 2, 0],
        }
    )
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"]
    return df


@pytest.fixture
def parquet_path(sample_df: pd.DataFrame, tmp_path: Path) -> str:
    """Write a Parquet feature store file from sample data."""
    path = str(tmp_path / "features.parquet")
    sample_df.to_parquet(path, index=False)
    return path


# ── load_from_feature_store ──────────────────────────────────────────────────


class TestLoadFromFeatureStore:
    def test_loads_correct_shape(self, parquet_path: str) -> None:
        df = load_from_feature_store(parquet_path)
        assert len(df) == 4

    def test_all_feature_cols_present(self, parquet_path: str) -> None:
        df = load_from_feature_store(parquet_path)
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_target_present(self, parquet_path: str) -> None:
        df = load_from_feature_store(parquet_path)
        assert TARGET_COL in df.columns


# ── Train and evaluate ───────────────────────────────────────────────────────


class TestTrainAndEvaluate:
    def test_returns_model_and_metrics(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[FEATURE_COLS]
        y = sample_df[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        model, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_metrics_keys(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[FEATURE_COLS]
        y = sample_df[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        assert "cv_mae_mean" in metrics
        assert "cv_mae_std" in metrics
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics

    def test_feature_importances_logged(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[FEATURE_COLS]
        y = sample_df[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for col in FEATURE_COLS:
            key = f"importance_{col}"
            assert key in metrics, f"Missing importance metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_metrics_are_finite(self, sample_df: pd.DataFrame) -> None:
        X = sample_df[FEATURE_COLS]
        y = sample_df[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for name, value in metrics.items():
            assert np.isfinite(value), f"Metric {name} is not finite: {value}"
