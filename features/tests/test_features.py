"""Tests for the feature pipeline — pure logic, no Feast/Redis needed."""

from pathlib import Path

import pandas as pd
import pytest
from pipeline import compute_features, load_csv_vehicles, write_parquet

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_and_compute(data_dir: Path) -> pd.DataFrame:
    """Load CSVs and compute features (no saved vehicles)."""
    vehicles, reservations = load_csv_vehicles(data_dir)
    return compute_features(vehicles, reservations, pd.DataFrame())


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create minimal CSV files for testing."""
    vehicles = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3],
            "technology": [1, 0, 1],
            "actual_price": [45.0, 30.0, 60.0],
            "recommended_price": [50.0, 25.0, 55.0],
            "num_images": [8, 2, 15],
            "street_parked": [0, 1, 0],
            "description": [250, 50, 400],
        }
    )
    reservations = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 2, 3, 3],
            "created_at": ["2025-01-01"] * 6,
        }
    )
    vehicles.to_csv(tmp_path / "vehicles.csv", index=False)
    reservations.to_csv(tmp_path / "reservations.csv", index=False)
    return tmp_path


# ── compute_features ─────────────────────────────────────────────────────────


class TestComputeFeatures:
    def test_returns_all_vehicles(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        assert len(df) == 3

    def test_reservation_counts(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        counts = df.set_index("vehicle_id")["num_reservations"]
        assert counts[1] == 3
        assert counts[2] == 1
        assert counts[3] == 2

    def test_derived_features_present(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        assert "price_diff" in df.columns
        assert "price_ratio" in df.columns

    def test_price_diff_values(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        row = df[df["vehicle_id"] == 1].iloc[0]
        assert row["price_diff"] == pytest.approx(-5.0)

    def test_price_ratio_values(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        row = df[df["vehicle_id"] == 1].iloc[0]
        assert row["price_ratio"] == pytest.approx(0.9)

    def test_event_timestamp_present(self, data_dir: Path) -> None:
        df = load_and_compute(data_dir)
        assert "event_timestamp" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"])

    def test_vehicles_with_no_reservations(self, tmp_path: Path) -> None:
        """Vehicle with no reservations should get num_reservations=0."""
        vehicles = pd.DataFrame(
            {
                "vehicle_id": [1, 2],
                "technology": [1, 0],
                "actual_price": [45.0, 30.0],
                "recommended_price": [50.0, 25.0],
                "num_images": [8, 2],
                "street_parked": [0, 1],
                "description": [250, 50],
            }
        )
        reservations = pd.DataFrame(
            {
                "vehicle_id": [1],
                "created_at": ["2025-01-01"],
            }
        )
        vehicles.to_csv(tmp_path / "vehicles.csv", index=False)
        reservations.to_csv(tmp_path / "reservations.csv", index=False)

        v, r = load_csv_vehicles(tmp_path)
        df = compute_features(v, r, pd.DataFrame())
        row = df[df["vehicle_id"] == 2].iloc[0]
        assert row["num_reservations"] == 0

    def test_includes_saved_vehicles(self, data_dir: Path) -> None:
        """Saved vehicles should be merged with CSV vehicles."""
        vehicles, reservations = load_csv_vehicles(data_dir)
        saved = pd.DataFrame(
            {
                "vehicle_id": [100],
                "technology": [1],
                "actual_price": [99.0],
                "recommended_price": [80.0],
                "num_images": [20],
                "street_parked": [0],
                "description": [500],
            }
        )
        df = compute_features(vehicles, reservations, saved)
        assert len(df) == 4  # 3 CSV + 1 saved
        saved_row = df[df["vehicle_id"] == 100].iloc[0]
        assert saved_row["num_reservations"] == 0
        assert saved_row["price_diff"] == pytest.approx(19.0)


# ── write_parquet ────────────────────────────────────────────────────────────


class TestWriteParquet:
    def test_creates_file(self, data_dir: Path, tmp_path: Path) -> None:
        df = load_and_compute(data_dir)
        out = str(tmp_path / "subdir" / "features.parquet")
        write_parquet(df, out)
        assert Path(out).exists()

    def test_roundtrip(self, data_dir: Path, tmp_path: Path) -> None:
        df = load_and_compute(data_dir)
        out = str(tmp_path / "features.parquet")
        write_parquet(df, out)
        loaded = pd.read_parquet(out)
        assert len(loaded) == len(df)
        assert set(loaded.columns) == set(df.columns)


# ── Feature definitions ──────────────────────────────────────────────────────


class TestDefinitions:
    def test_feature_refs_match_schema(self) -> None:
        from feature_repo.definitions import FEATURE_REFS, vehicle_features_view

        schema_names = {f.name for f in vehicle_features_view.schema}
        ref_names = {ref.split(":")[1] for ref in FEATURE_REFS}
        assert ref_names.issubset(schema_names)

    def test_label_ref_in_schema(self) -> None:
        from feature_repo.definitions import LABEL_REF, vehicle_features_view

        schema_names = {f.name for f in vehicle_features_view.schema}
        label_name = LABEL_REF.split(":")[1]
        assert label_name in schema_names

    def test_label_not_in_feature_refs(self) -> None:
        from feature_repo.definitions import FEATURE_REFS, LABEL_REF

        label_name = LABEL_REF.split(":")[1]
        ref_names = {ref.split(":")[1] for ref in FEATURE_REFS}
        assert label_name not in ref_names, "Label should not be in FEATURE_REFS"
