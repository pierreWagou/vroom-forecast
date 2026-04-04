"""
Vroom Forecast — Feature Pipeline

Single source of truth for feature computation. Reads all vehicles and
reservations from the SQLite database, computes derived features, writes
to the Feast offline store (Parquet), and materializes to the online
store (Redis).

The database must be seeded first (see seed.py).

Usage:
    uv run python pipeline.py --db /feast-data/vehicles.db --feast-repo feature_repo
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
DB_PATH = "/feast-data/vehicles.db"


def load_from_db(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all vehicles and reservations from the SQLite database."""
    conn = sqlite3.connect(db_path)
    vehicles = pd.read_sql_query(
        "SELECT vehicle_id, technology, actual_price, recommended_price, "
        "num_images, street_parked, description FROM vehicles",
        conn,
    )
    reservations = pd.read_sql_query("SELECT vehicle_id, created_at FROM reservations", conn)
    conn.close()
    logger.info(
        "Loaded %d vehicles and %d reservations from database.", len(vehicles), len(reservations)
    )
    return vehicles, reservations


def compute_features(vehicles: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reservations and compute derived features."""
    # Aggregate reservation counts per vehicle
    res_counts = (
        reservations.groupby("vehicle_id").size().reset_index(name="num_reservations")  # ty: ignore[no-matching-overload]
    )

    df = vehicles.merge(res_counts, on="vehicle_id", how="left")
    df["num_reservations"] = df["num_reservations"].fillna(0).astype(int)

    # Derived features — the ONLY place these are computed
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"].replace(0, float("nan"))

    # Feast requires a timestamp column
    df["event_timestamp"] = pd.Timestamp(datetime.now(tz=UTC))

    logger.info("Computed features for %d vehicles.", len(df))
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


def run(db_path: str, feast_repo: str, parquet_path: str) -> None:
    """Full feature pipeline: load → compute → write → apply → materialize."""
    vehicles, reservations = load_from_db(db_path)
    df = compute_features(vehicles, reservations)
    write_parquet(df, parquet_path)
    apply_and_materialize(feast_repo)
    logger.info("Feature pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature pipeline")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to SQLite database")
    parser.add_argument(
        "--feast-repo", type=str, default="feature_repo", help="Path to Feast feature repo"
    )
    parser.add_argument(
        "--parquet-path", type=str, default=PARQUET_PATH, help="Output Parquet path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(db_path=args.db, feast_repo=args.feast_repo, parquet_path=args.parquet_path)
