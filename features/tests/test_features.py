"""Tests for the feature pipeline — pure logic, no Feast/Redis needed."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from pipeline import compute_features, load_from_db, write_parquet
from seed import init_db, seed

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


@pytest.fixture
def seeded_db(data_dir: Path, tmp_path: Path) -> str:
    """Seed a SQLite database from the test CSVs."""
    db_path = str(tmp_path / "test.db")
    seed(data_dir=data_dir, db_path=db_path)
    return db_path


# ── seed ─────────────────────────────────────────────────────────────────────


class TestSeed:
    def test_seeds_vehicles(self, seeded_db: str) -> None:
        conn = sqlite3.connect(seeded_db)
        count = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        conn.close()
        assert count == 3

    def test_seeds_reservations(self, seeded_db: str) -> None:
        conn = sqlite3.connect(seeded_db)
        count = conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]
        conn.close()
        assert count == 6

    def test_idempotent(self, data_dir: Path, seeded_db: str) -> None:
        """Running seed twice should not duplicate data."""
        seed(data_dir=data_dir, db_path=seeded_db)
        conn = sqlite3.connect(seeded_db)
        count = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        conn.close()
        assert count == 3

    def test_source_column(self, seeded_db: str) -> None:
        conn = sqlite3.connect(seeded_db)
        sources = conn.execute("SELECT DISTINCT source FROM vehicles").fetchall()
        conn.close()
        assert sources == [("csv",)]


# ── compute_features ─────────────────────────────────────────────────────────


class TestComputeFeatures:
    def test_returns_all_vehicles(self, seeded_db: str) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        assert len(df) == 3

    def test_reservation_counts(self, seeded_db: str) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        counts = df.set_index("vehicle_id")["num_reservations"]
        assert counts[1] == 3
        assert counts[2] == 1
        assert counts[3] == 2

    def test_derived_features_present(self, seeded_db: str) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        assert "price_diff" in df.columns
        assert "price_ratio" not in df.columns  # dropped — collinear with price_diff
        assert "actual_price" not in df.columns  # dropped — not a model feature
        assert "recommended_price" not in df.columns  # dropped — not a model feature

    def test_price_diff_values(self, seeded_db: str) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        row = df[df["vehicle_id"] == 1].iloc[0]
        assert row["price_diff"] == pytest.approx(-5.0)

    def test_event_timestamp_present(self, seeded_db: str) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        assert "event_timestamp" in df.columns

    def test_vehicles_with_no_reservations(self, tmp_path: Path) -> None:
        """Fleet vehicle (csv) with no reservations should get num_reservations=0."""
        db_path = str(tmp_path / "test2.db")
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO vehicles (technology, actual_price, recommended_price, "
            "num_images, street_parked, description, source) VALUES (1, 45, 50, 8, 0, 250, 'csv')"
        )
        conn.commit()
        conn.close()

        vehicles, reservations = load_from_db(db_path)
        df = compute_features(vehicles, reservations)
        assert df.iloc[0]["num_reservations"] == 0

    def test_new_arrival_has_null_reservations(self, tmp_path: Path) -> None:
        """UI vehicle with no reservations should get num_reservations=NULL."""
        db_path = str(tmp_path / "test3.db")
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO vehicles (technology, actual_price, recommended_price, "
            "num_images, street_parked, description, source) VALUES (1, 45, 50, 8, 0, 250, 'ui')"
        )
        conn.commit()
        conn.close()

        vehicles, reservations = load_from_db(db_path)
        df = compute_features(vehicles, reservations)
        assert pd.isna(df.iloc[0]["num_reservations"])


# ── write_parquet ────────────────────────────────────────────────────────────


class TestWriteParquet:
    def test_creates_file(self, seeded_db: str, tmp_path: Path) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
        out = str(tmp_path / "subdir" / "features.parquet")
        write_parquet(df, out)
        assert Path(out).exists()

    def test_roundtrip(self, seeded_db: str, tmp_path: Path) -> None:
        vehicles, reservations = load_from_db(seeded_db)
        df = compute_features(vehicles, reservations)
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
