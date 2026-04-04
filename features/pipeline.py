"""
Vroom Forecast — Feature Pipeline

Single source of truth for feature computation. Reads raw data from CSVs
and user-saved vehicles from SQLite, computes all features, writes to the
Feast offline store (Parquet), and materializes to the online store (Redis).

Usage:
    uv run python pipeline.py \
        --data-dir data \
        --feast-repo features/feature_repo \
        [--vehicles-db /feast-data/vehicles.db]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from feature_repo.definitions import vehicle, vehicle_features_source, vehicle_features_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = "/feast-data/vehicle_features.parquet"
VEHICLES_DB = "/feast-data/vehicles.db"


def load_csv_vehicles(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw vehicles and reservations from CSVs."""
    vehicles = pd.read_csv(data_dir / "vehicles.csv")
    reservations = pd.read_csv(data_dir / "reservations.csv")
    logger.info(
        "Loaded %d vehicles and %d reservations from CSV.", len(vehicles), len(reservations)
    )
    return vehicles, reservations


def load_saved_vehicles(db_path: str) -> pd.DataFrame:
    """Load user-saved vehicles from the SQLite database."""
    path = Path(db_path)
    if not path.exists():
        logger.info("No vehicles database at %s — skipping.", db_path)
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT vehicle_id, technology, actual_price, recommended_price, "
        "num_images, street_parked, description FROM vehicles",
        conn,
    )
    conn.close()

    if len(df) == 0:
        logger.info("Vehicles database is empty — skipping.")
        return df

    # Offset IDs to avoid collision with CSV vehicle IDs.
    # CSV vehicles use IDs starting from 1. User-saved vehicles start from 100_000.
    df["vehicle_id"] = df["vehicle_id"] + 100_000
    logger.info("Loaded %d user-saved vehicles from SQLite.", len(df))
    return df


def compute_features(
    vehicles: pd.DataFrame, reservations: pd.DataFrame, saved_vehicles: pd.DataFrame
) -> pd.DataFrame:
    """Merge all vehicle sources, aggregate reservations, and compute derived features."""
    # Combine CSV vehicles and user-saved vehicles
    all_vehicles = pd.concat([vehicles, saved_vehicles], ignore_index=True)

    # Aggregate reservation counts per vehicle (only CSV vehicles have reservations)
    res_counts = (
        reservations.groupby("vehicle_id").size().reset_index(name="num_reservations")  # ty: ignore[no-matching-overload]
    )

    df = all_vehicles.merge(res_counts, on="vehicle_id", how="left")
    df["num_reservations"] = df["num_reservations"].fillna(0).astype(int)

    # Derived features — the ONLY place these are computed
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"]

    # Feast requires a timestamp column
    df["event_timestamp"] = pd.Timestamp(datetime.now(tz=UTC))

    logger.info(
        "Computed features for %d vehicles (%d from CSV, %d user-saved).",
        len(df),
        len(vehicles),
        len(saved_vehicles),
    )
    return df


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write the feature DataFrame to Parquet for Feast's offline store."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def apply_and_materialize(feast_repo: str) -> None:
    """Apply the Feast repo and materialize features to the online store."""
    store = FeatureStore(repo_path=feast_repo)
    store.apply(
        [vehicle, vehicle_features_source, vehicle_features_view]  # type: ignore[list-item]
    )
    logger.info("Feast repo applied.")

    store.materialize(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime.now(tz=UTC),
    )
    logger.info("Features materialized to online store.")


def run(data_dir: Path, feast_repo: str, parquet_path: str, vehicles_db: str) -> None:
    """Full feature pipeline: load → compute → write → apply → materialize."""
    vehicles, reservations = load_csv_vehicles(data_dir)
    saved = load_saved_vehicles(vehicles_db)
    df = compute_features(vehicles, reservations, saved)
    write_parquet(df, parquet_path)
    apply_and_materialize(feast_repo)
    logger.info("Feature pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing vehicles.csv and reservations.csv",
    )
    parser.add_argument(
        "--feast-repo",
        type=str,
        default="feature_repo",
        help="Path to the Feast feature repo directory",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=PARQUET_PATH,
        help="Output path for the Parquet file",
    )
    parser.add_argument(
        "--vehicles-db",
        type=str,
        default=VEHICLES_DB,
        help="Path to the SQLite database with user-saved vehicles",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        feast_repo=args.feast_repo,
        parquet_path=args.parquet_path,
        vehicles_db=args.vehicles_db,
    )
