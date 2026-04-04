"""Tests for the serving API — feature engineering, schemas, and API contract.

Note: Full endpoint integration tests require a running Ray Serve cluster.
These tests focus on unit-testable components: schemas, feature engineering.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from serving.features import FEATURE_COLS, engineer_features
from serving.schemas import VehicleFeatures

SAMPLE_VEHICLE = {
    "technology": 1,
    "actual_price": 45.0,
    "recommended_price": 50.0,
    "num_images": 8,
    "street_parked": 0,
    "description": 250,
}


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


# ── Schema contract ──────────────────────────────────────────────────────────


class TestSchemaContract:
    """Verify that schemas have the expected fields — catches drift with the UI."""

    def test_prediction_response_fields(self) -> None:
        from serving.schemas import PredictionResponse

        fields = set(PredictionResponse.model_fields.keys())
        assert {"predicted_reservations", "model_version"} == fields

    def test_health_response_fields(self) -> None:
        from serving.schemas import HealthResponse

        fields = set(HealthResponse.model_fields.keys())
        assert {"status", "model_name", "model_version", "mlflow_uri", "feast_online"} == fields

    def test_benchmark_response_fields(self) -> None:
        from serving.schemas import BenchmarkResponse

        fields = set(BenchmarkResponse.model_fields.keys())
        expected = {
            "n_iterations",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "model_version",
            "source",
            "avg_features_ms",
            "avg_predict_ms",
        }
        assert expected == fields

    def test_computed_features_fields(self) -> None:
        from serving.schemas import ComputedFeatures

        fields = set(ComputedFeatures.model_fields.keys())
        assert "materialized" in fields
        assert "price_diff" in fields
        assert "price_ratio" in fields
