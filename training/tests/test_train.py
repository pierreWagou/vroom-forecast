"""Tests for the training pipeline — pure logic, no MLflow needed."""

import numpy as np
import pandas as pd
import pytest
from train import (
    FEATURE_COLS,
    TARGET_COL,
    engineer_features,
    train_and_evaluate,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def raw_vehicles() -> pd.DataFrame:
    """Minimal vehicle dataset before feature engineering."""
    return pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "technology": [1, 0, 1, 0],
            "actual_price": [45.0, 30.0, 60.0, 25.0],
            "recommended_price": [50.0, 25.0, 55.0, 30.0],
            "num_images": [8, 2, 15, 0],
            "street_parked": [0, 1, 0, 1],
            "description": [250, 50, 400, 10],
            TARGET_COL: [10, 3, 20, 1],
        }
    )


@pytest.fixture
def engineered(raw_vehicles: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(raw_vehicles)


# ── Feature engineering ──────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_adds_price_diff(self, engineered: pd.DataFrame) -> None:
        expected = engineered["actual_price"] - engineered["recommended_price"]
        pd.testing.assert_series_equal(engineered["price_diff"], expected, check_names=False)

    def test_adds_price_ratio(self, engineered: pd.DataFrame) -> None:
        expected = engineered["actual_price"] / engineered["recommended_price"]
        pd.testing.assert_series_equal(engineered["price_ratio"], expected, check_names=False)

    def test_does_not_mutate_input(self, raw_vehicles: pd.DataFrame) -> None:
        original_cols = set(raw_vehicles.columns)
        engineer_features(raw_vehicles)
        assert set(raw_vehicles.columns) == original_cols

    def test_all_feature_cols_present(self, engineered: pd.DataFrame) -> None:
        for col in FEATURE_COLS:
            assert col in engineered.columns, f"Missing feature column: {col}"


# ── Train and evaluate ───────────────────────────────────────────────────────


class TestTrainAndEvaluate:
    def test_returns_model_and_metrics(self, engineered: pd.DataFrame) -> None:
        X = engineered[FEATURE_COLS]
        y = engineered[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        model, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        # Model is fitted
        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_metrics_keys(self, engineered: pd.DataFrame) -> None:
        X = engineered[FEATURE_COLS]
        y = engineered[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        assert "cv_mae_mean" in metrics
        assert "cv_mae_std" in metrics
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics

    def test_feature_importances_logged(self, engineered: pd.DataFrame) -> None:
        X = engineered[FEATURE_COLS]
        y = engineered[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for col in FEATURE_COLS:
            key = f"importance_{col}"
            assert key in metrics, f"Missing importance metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_metrics_are_finite(self, engineered: pd.DataFrame) -> None:
        X = engineered[FEATURE_COLS]
        y = engineered[TARGET_COL]
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

        _, metrics = train_and_evaluate(X, y, params, cv_folds=2)

        for name, value in metrics.items():
            assert np.isfinite(value), f"Metric {name} is not finite: {value}"
