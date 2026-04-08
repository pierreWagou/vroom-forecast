"""Tests for the serving API — feature engineering, schemas, and API contract.

Note: Full endpoint integration tests require a running Ray Serve cluster.
These tests focus on unit-testable components: schemas, feature engineering.
"""

import os

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
        assert "store" in fields

    def test_computed_features_store_literal(self) -> None:
        from serving.schemas import ComputedFeatures

        feat = ComputedFeatures(vehicle_id=1, store="offline")
        assert feat.store == "offline"
        feat = ComputedFeatures(vehicle_id=1, store="online")
        assert feat.store == "online"
        feat = ComputedFeatures(vehicle_id=1)
        assert feat.store == "none"

    def test_save_vehicle_response_fields(self) -> None:
        from serving.schemas import SaveVehicleResponse

        fields = set(SaveVehicleResponse.model_fields.keys())
        assert {"vehicle_id", "status", "event_published"} == fields

    def test_vehicle_record_fields(self) -> None:
        from serving.schemas import VehicleRecord

        fields = set(VehicleRecord.model_fields.keys())
        expected = {
            "vehicle_id",
            "technology",
            "actual_price",
            "recommended_price",
            "num_images",
            "street_parked",
            "description",
            "source",
            "num_reservations",
        }
        assert expected == fields

    def test_vehicle_record_nullable_reservations(self) -> None:
        from serving.schemas import VehicleRecord

        # Fleet vehicle: observed count (including 0)
        fleet = VehicleRecord(
            vehicle_id=1,
            technology=1,
            actual_price=45,
            recommended_price=50,
            num_images=5,
            street_parked=0,
            description=100,
            num_reservations=0,
        )
        assert fleet.num_reservations == 0

        # New arrival: no observation
        arrival = VehicleRecord(
            vehicle_id=2,
            technology=1,
            actual_price=45,
            recommended_price=50,
            num_images=5,
            street_parked=0,
            description=100,
        )
        assert arrival.num_reservations is None

    def test_benchmark_by_id_request_fields(self) -> None:
        from serving.schemas import BenchmarkByIdRequest

        fields = set(BenchmarkByIdRequest.model_fields.keys())
        assert {"n_iterations", "vehicle_id"} == fields


# ── Feature engineering: benchmark helper ────────────────────────────────────


class TestEngineerFeaturesBenchmark:
    """Test that engineer_features works correctly in a loop (benchmark path)."""

    def test_repeated_compute_is_deterministic(self) -> None:
        df = pd.DataFrame([SAMPLE_VEHICLE])
        results = [engineer_features(df)["price_diff"].iloc[0] for _ in range(10)]
        assert all(r == pytest.approx(-5.0) for r in results)


# ── Offline feature reader ───────────────────────────────────────────────────


class TestOfflineFeatureReader:
    """Test the offline feature reading logic with a real temporary Parquet file.

    OfflineFeatureReader is a @serve.deployment so we can't instantiate it
    directly in tests. Instead we test the underlying logic: loading a Parquet
    file and looking up vehicle IDs from the resulting DataFrame.
    """

    @pytest.fixture()
    def parquet_df(self, tmp_path: object) -> tuple[str, pd.DataFrame]:
        import os

        df = pd.DataFrame(
            [
                {
                    "vehicle_id": 1,
                    "technology": 1,
                    "num_images": 8,
                    "street_parked": 0,
                    "description": 250,
                    "price_diff": -5.0,
                    "num_reservations": 5,
                    "event_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "vehicle_id": 2,
                    "technology": 0,
                    "num_images": 3,
                    "street_parked": 1,
                    "description": 100,
                    "price_diff": 50.0,
                    "num_reservations": 12,
                    "event_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                },
            ]
        )
        path = os.path.join(str(tmp_path), "features.parquet")
        df.to_parquet(path, index=False)
        indexed = df.set_index("vehicle_id")
        return path, indexed

    def test_lookup_found(self, parquet_df: tuple[str, pd.DataFrame]) -> None:
        _, df = parquet_df
        mask = df.index.isin([1])
        assert mask.any()
        result = df.loc[mask].reset_index()
        assert len(result) == 1
        assert result.iloc[0]["vehicle_id"] == 1
        assert result.iloc[0]["price_diff"] == pytest.approx(-5.0)

    def test_lookup_not_found(self, parquet_df: tuple[str, pd.DataFrame]) -> None:
        _, df = parquet_df
        mask = df.index.isin([999])
        assert not mask.any()

    def test_lookup_partial_match(self, parquet_df: tuple[str, pd.DataFrame]) -> None:
        _, df = parquet_df
        mask = df.index.isin([1, 999])
        result = df.loc[mask].reset_index()
        assert len(result) == 1

    def test_parquet_loads_correctly(self, parquet_df: tuple[str, pd.DataFrame]) -> None:
        path, _ = parquet_df
        loaded = pd.read_parquet(path).set_index("vehicle_id")
        assert len(loaded) == 2
        assert 1 in loaded.index
        assert 2 in loaded.index

    def test_empty_parquet(self, tmp_path: object) -> None:
        import os

        df = pd.DataFrame(
            columns=[
                "vehicle_id",
                "technology",
                "num_images",
                "street_parked",
                "description",
                "price_diff",
                "num_reservations",
                "event_timestamp",
            ]
        )
        path = os.path.join(str(tmp_path), "empty.parquet")
        df.to_parquet(path, index=False)
        loaded = pd.read_parquet(path).set_index("vehicle_id")
        mask = loaded.index.isin([1])
        assert not mask.any()


# ── Vehicles CRUD ────────────────────────────────────────────────────────────


class TestVehicles:
    """Test vehicle persistence (SQLite CRUD) without Redis."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point DB_PATH to a temporary database and disable Redis."""
        db = os.path.join(str(tmp_path), "test_vehicles.db")
        monkeypatch.setattr("serving.vehicles.DB_PATH", db)
        monkeypatch.setattr("serving.config.settings.redis_url", None)

    def test_save_and_list(self) -> None:
        from serving.vehicles import list_vehicles, save_vehicle

        v = VehicleFeatures(
            technology=1,
            actual_price=45.0,
            recommended_price=50.0,
            num_images=8,
            street_parked=0,
            description=250,
        )
        vid, published = save_vehicle(v)
        assert vid > 0
        assert published is False  # no Redis

        vehicles = list_vehicles()
        assert len(vehicles) == 1
        assert vehicles[0].vehicle_id == vid
        assert vehicles[0].source == "ui"
        assert vehicles[0].technology == 1
        assert vehicles[0].num_reservations is None  # UI vehicle → no observation

    def test_delete_ui_vehicle(self) -> None:
        from serving.vehicles import delete_vehicle, list_vehicles, save_vehicle

        v = VehicleFeatures(
            technology=0,
            actual_price=30.0,
            recommended_price=25.0,
            num_images=2,
            street_parked=1,
            description=50,
        )
        vid, _ = save_vehicle(v)
        assert delete_vehicle(vid) is True
        assert list_vehicles() == []

    def test_delete_nonexistent_returns_false(self) -> None:
        from serving.vehicles import delete_vehicle

        assert delete_vehicle(99999) is False

    def test_delete_csv_vehicle_blocked(self, tmp_path: object) -> None:
        """CSV-sourced vehicles cannot be deleted via delete_vehicle."""
        from serving.vehicles import _get_db, delete_vehicle

        conn = _get_db()
        conn.execute(
            "INSERT INTO vehicles (technology, actual_price, recommended_price, "
            "num_images, street_parked, description, source) "
            "VALUES (1, 45, 50, 8, 0, 250, 'csv')"
        )
        conn.commit()
        vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        assert delete_vehicle(vid) is False

    def test_list_with_reservations_table(self, tmp_path: object) -> None:
        """list_vehicles handles the JOIN when reservations table exists."""
        from serving.vehicles import _get_db, list_vehicles

        # Insert a CSV vehicle with reservations
        conn = _get_db()
        conn.execute(
            "INSERT INTO vehicles (technology, actual_price, recommended_price, "
            "num_images, street_parked, description, source) "
            "VALUES (1, 45, 50, 8, 0, 250, 'csv')"
        )
        vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reservations "
            "(id INTEGER PRIMARY KEY, vehicle_id INTEGER, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO reservations (vehicle_id, created_at) VALUES (?, '2025-01-01')",
            (vid,),
        )
        conn.execute(
            "INSERT INTO reservations (vehicle_id, created_at) VALUES (?, '2025-01-02')",
            (vid,),
        )
        conn.commit()
        conn.close()

        vehicles = list_vehicles()
        assert len(vehicles) == 1
        assert vehicles[0].num_reservations == 2
        assert vehicles[0].source == "csv"

    def test_multiple_vehicles_ordered(self) -> None:
        from serving.vehicles import list_vehicles, save_vehicle

        for price in [10, 20, 30]:
            save_vehicle(
                VehicleFeatures(
                    technology=0,
                    actual_price=float(price),
                    recommended_price=25.0,
                    num_images=1,
                    street_parked=0,
                    description=10,
                )
            )

        vehicles = list_vehicles()
        assert len(vehicles) == 3
        # Ordered by vehicle_id (ascending)
        assert vehicles[0].vehicle_id < vehicles[1].vehicle_id < vehicles[2].vehicle_id


# ── Config ───────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_values(self) -> None:
        from serving.config import Settings

        s = Settings()
        assert s.mlflow_uri == "http://localhost:5001"
        assert s.model_name == "vroom-forecast"
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.feast_repo is None
        assert s.redis_url is None
        assert s.db_path == "/feast-data/vehicles.db"
        assert s.offline_store_path is None

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from serving.config import Settings

        monkeypatch.setenv("SERVING_MLFLOW_URI", "http://custom:9000")
        monkeypatch.setenv("SERVING_PORT", "9090")
        monkeypatch.setenv("SERVING_REDIS_URL", "redis://myredis:6379")
        s = Settings()
        assert s.mlflow_uri == "http://custom:9000"
        assert s.port == 9090
        assert s.redis_url == "redis://myredis:6379"
