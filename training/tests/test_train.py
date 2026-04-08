"""Tests for the training pipeline — pure logic, no MLflow needed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from train import (
    FEATURE_COLS,
    MODEL_NAME,
    TARGET_COL,
    load_from_feature_store,
    parse_args,
    register_model,
    run,
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


# ── register_model ───────────────────────────────────────────────────────────


class TestRegisterModel:
    @patch("train.mlflow")
    def test_creates_model_and_sets_alias(self, mock_mlflow: MagicMock) -> None:
        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mv = MagicMock()
        mv.version = "1"
        client.create_model_version.return_value = mv

        version = register_model("run-123", "vroom-forecast")

        assert version == "1"
        client.get_registered_model.assert_called_once_with("vroom-forecast")
        client.create_model_version.assert_called_once_with(
            name="vroom-forecast", source="runs:/run-123/model", run_id="run-123"
        )
        client.set_registered_model_alias.assert_called_once_with(
            "vroom-forecast", "candidate", "1"
        )

    @patch("train.mlflow")
    def test_creates_registered_model_if_missing(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        exc = MlflowException("Not found")
        exc.error_code = "RESOURCE_DOES_NOT_EXIST"
        client.get_registered_model.side_effect = exc
        mv = MagicMock()
        mv.version = "1"
        client.create_model_version.return_value = mv

        version = register_model("run-123", "my-model")

        assert version == "1"
        client.create_registered_model.assert_called_once_with("my-model")

    @patch("train.mlflow")
    def test_reraises_unexpected_mlflow_error(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        exc = MlflowException("Network error")
        exc.error_code = "INTERNAL_ERROR"
        client.get_registered_model.side_effect = exc

        with pytest.raises(MlflowException):
            register_model("run-123", "vroom-forecast")


# ── run (full pipeline) ─────────────────────────────────────────────────────


class TestRun:
    @patch("train.train_and_evaluate")
    @patch("train.register_model")
    @patch("train.mlflow")
    def test_full_pipeline(
        self,
        mock_mlflow: MagicMock,
        mock_register: MagicMock,
        mock_train_eval: MagicMock,
        parquet_path: str,
    ) -> None:
        # Mock train_and_evaluate to return a fake model and metrics
        fake_model = MagicMock()
        fake_metrics = {"cv_mae_mean": 1.5, "train_mae": 1.0}
        mock_train_eval.return_value = (fake_model, fake_metrics)
        mock_register.return_value = "7"

        # Mock mlflow.active_run() to return a run with an ID
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc"
        mock_mlflow.active_run.return_value = mock_run

        version = run(
            mlflow_uri="http://fake:5000",
            experiment_name="test-exp",
            model_name="test-model",
            feature_store=parquet_path,
        )

        assert version == "7"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://fake:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()
        mock_register.assert_called_once_with("run-abc", "test-model")

    @patch("train.train_and_evaluate")
    @patch("train.register_model")
    @patch("train.mlflow")
    def test_raises_if_no_active_run(
        self,
        mock_mlflow: MagicMock,
        mock_register: MagicMock,
        mock_train_eval: MagicMock,
        parquet_path: str,
    ) -> None:
        mock_train_eval.return_value = (MagicMock(), {"cv_mae_mean": 1.5})
        mock_mlflow.active_run.return_value = None

        with pytest.raises(RuntimeError, match="No active MLflow run"):
            run(
                mlflow_uri="http://fake:5000",
                experiment_name="test",
                model_name="test",
                feature_store=parquet_path,
            )


# ── parse_args ───────────────────────────────────────────────────────────────


class TestParseArgs:
    def test_required_feature_store(self) -> None:
        with pytest.raises(SystemExit):
            parse_args()  # --feature-store is required

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["train", "--feature-store", "features.parquet"])
        args = parse_args()
        assert args.feature_store == "features.parquet"
        assert args.mlflow_uri == "http://localhost:5001"
        assert args.experiment == "vroom-forecast"
        assert args.model_name == MODEL_NAME

    def test_custom_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "train",
                "--feature-store",
                "custom.parquet",
                "--mlflow-uri",
                "http://my-mlflow:5000",
                "--experiment",
                "my-exp",
                "--model-name",
                "my-model",
            ],
        )
        args = parse_args()
        assert args.feature_store == "custom.parquet"
        assert args.mlflow_uri == "http://my-mlflow:5000"
        assert args.experiment == "my-exp"
        assert args.model_name == "my-model"
