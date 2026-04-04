"""Tests for the training pipeline — pure logic, no MLflow needed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from train import (
    FEATURE_COLS,
    TARGET_COL,
    load_from_csv,
    load_from_feature_store,
    train_and_evaluate,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create minimal CSV files for testing."""
    vehicles = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "technology": [1, 0, 1, 0],
            "actual_price": [45.0, 30.0, 60.0, 25.0],
            "recommended_price": [50.0, 25.0, 55.0, 30.0],
            "num_images": [8, 2, 15, 0],
            "street_parked": [0, 1, 0, 1],
            "description": [250, 50, 400, 10],
        }
    )
    reservations = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 2, 3, 3],
            "created_at": pd.to_datetime(["2025-01-01"] * 6),
        }
    )
    vehicles.to_csv(tmp_path / "vehicles.csv", index=False)
    reservations.to_csv(tmp_path / "reservations.csv", index=False)
    return tmp_path


@pytest.fixture
def df_from_csv(data_dir: Path) -> pd.DataFrame:
    return load_from_csv(data_dir)


@pytest.fixture
def parquet_path(df_from_csv: pd.DataFrame, tmp_path: Path) -> str:
    """Write a Parquet feature store file from the CSV data."""
    path = str(tmp_path / "features.parquet")
    df_from_csv.to_parquet(path, index=False)
    return path


# ── load_from_csv ────────────────────────────────────────────────────────────


class TestLoadFromCsv:
    def test_all_feature_cols_present(self, df_from_csv: pd.DataFrame) -> None:
        for col in FEATURE_COLS:
            assert col in df_from_csv.columns, f"Missing feature column: {col}"

    def test_target_col_present(self, df_from_csv: pd.DataFrame) -> None:
        assert TARGET_COL in df_from_csv.columns

    def test_price_diff(self, df_from_csv: pd.DataFrame) -> None:
        expected = df_from_csv["actual_price"] - df_from_csv["recommended_price"]
        pd.testing.assert_series_equal(df_from_csv["price_diff"], expected, check_names=False)

    def test_price_ratio(self, df_from_csv: pd.DataFrame) -> None:
        expected = df_from_csv["actual_price"] / df_from_csv["recommended_price"]
        pd.testing.assert_series_equal(df_from_csv["price_ratio"], expected, check_names=False)

    def test_zero_reservations_filled(self, df_from_csv: pd.DataFrame) -> None:
        # Vehicle 4 has no reservations
        row = df_from_csv[df_from_csv["vehicle_id"] == 4].iloc[0]
        assert row[TARGET_COL] == 0


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
    def test_returns_model_and_metrics(self, df_from_csv: pd.DataFrame) -> None:
        X = df_from_csv[FEATURE_COLS]
        y = df_from_csv[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        model, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_metrics_keys(self, df_from_csv: pd.DataFrame) -> None:
        X = df_from_csv[FEATURE_COLS]
        y = df_from_csv[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        assert "cv_mae_mean" in metrics
        assert "cv_mae_std" in metrics
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics

    def test_feature_importances_logged(self, df_from_csv: pd.DataFrame) -> None:
        X = df_from_csv[FEATURE_COLS]
        y = df_from_csv[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for col in FEATURE_COLS:
            key = f"importance_{col}"
            assert key in metrics, f"Missing importance metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_metrics_are_finite(self, df_from_csv: pd.DataFrame) -> None:
        X = df_from_csv[FEATURE_COLS]
        y = df_from_csv[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for name, value in metrics.items():
            assert np.isfinite(value), f"Metric {name} is not finite: {value}"
