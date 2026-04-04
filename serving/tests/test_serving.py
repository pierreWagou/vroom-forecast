"""Tests for the serving API — feature engineering, schemas, and endpoints."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from serving.features import FEATURE_COLS, engineer_features
from serving.schemas import VehicleFeatures

# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_VEHICLE = {
    "technology": 1,
    "actual_price": 45.0,
    "recommended_price": 50.0,
    "num_images": 8,
    "street_parked": 0,
    "description": 250,
}


@pytest.fixture
def mock_model() -> MagicMock:
    """A fake sklearn model that returns deterministic predictions."""
    model = MagicMock()
    model.predict.return_value = np.array([7.42])
    return model


@pytest.fixture
def client(mock_model: MagicMock) -> Generator[TestClient, None, None]:
    """TestClient with a mocked model (no MLflow needed)."""
    with (
        patch("serving.model._model", mock_model),
        patch("serving.model._model_version", "99"),
        patch("serving.app.load_champion"),
    ):
        from serving.app import app

        with TestClient(app) as c:
            yield c


# ── Feature engineering ──────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_price_diff(self) -> None:
        df = pd.DataFrame([SAMPLE_VEHICLE])
        result = engineer_features(df)
        assert result["price_diff"].iloc[0] == pytest.approx(-5.0)

    def test_price_ratio(self) -> None:
        df = pd.DataFrame([SAMPLE_VEHICLE])
        result = engineer_features(df)
        assert result["price_ratio"].iloc[0] == pytest.approx(0.9)

    def test_all_feature_cols_present(self) -> None:
        df = pd.DataFrame([SAMPLE_VEHICLE])
        result = engineer_features(df)
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame([SAMPLE_VEHICLE])
        original_cols = set(df.columns)
        engineer_features(df)
        assert set(df.columns) == original_cols


# ── Schema validation ────────────────────────────────────────────────────────


class TestVehicleFeatures:
    def test_valid_vehicle(self) -> None:
        v = VehicleFeatures(**SAMPLE_VEHICLE)  # type: ignore[arg-type]
        assert v.technology == 1
        assert v.actual_price == 45.0

    def test_invalid_technology(self) -> None:
        with pytest.raises(ValidationError):
            VehicleFeatures(**{**SAMPLE_VEHICLE, "technology": 2})  # type: ignore[arg-type]

    def test_negative_price(self) -> None:
        with pytest.raises(ValidationError):
            VehicleFeatures(**{**SAMPLE_VEHICLE, "actual_price": -10})  # type: ignore[arg-type]

    def test_zero_price(self) -> None:
        with pytest.raises(ValidationError):
            VehicleFeatures(**{**SAMPLE_VEHICLE, "actual_price": 0})  # type: ignore[arg-type]


# ── API endpoints ────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "99"


class TestPredictEndpoint:
    def test_predict_single(self, client: TestClient) -> None:
        resp = client.post("/predict", json=SAMPLE_VEHICLE)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_reservations" in data
        assert data["model_version"] == "99"
        assert isinstance(data["predicted_reservations"], float)

    def test_predict_invalid_input(self, client: TestClient) -> None:
        resp = client.post("/predict", json={"technology": 1})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client: TestClient, mock_model: MagicMock) -> None:
        mock_model.predict.return_value = np.array([7.42, 5.0])
        resp = client.post("/predict/batch", json=[SAMPLE_VEHICLE, SAMPLE_VEHICLE])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2

    def test_batch_empty(self, client: TestClient, mock_model: MagicMock) -> None:
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 0


class TestBenchmarkEndpoint:
    def test_benchmark(self, client: TestClient) -> None:
        resp = client.post(
            "/benchmark",
            json={"n_iterations": 10, "vehicle": SAMPLE_VEHICLE},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_iterations"] == 10
        assert data["avg_latency_ms"] > 0
        assert data["p50_latency_ms"] > 0
        assert data["p95_latency_ms"] > 0
        assert data["p99_latency_ms"] > 0
        assert data["model_version"] == "99"
