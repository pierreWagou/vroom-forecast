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
import contextlib
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
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        vehicles = pd.read_sql_query(
            "SELECT vehicle_id, technology, actual_price, recommended_price, "
            "num_images, street_parked, description, source FROM vehicles",
            conn,
        )
        reservations = pd.read_sql_query("SELECT vehicle_id, created_at FROM reservations", conn)
    logger.info(
        "Loaded %d vehicles and %d reservations from database.", len(vehicles), len(reservations)
    )
    return vehicles, reservations


def compute_features(vehicles: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reservations and compute derived features.

    For csv-sourced vehicles, num_reservations is the observed count (0 if no
    bookings — a valid data point). For ui-sourced vehicles, num_reservations
    is NULL (no observation yet — needs prediction).
    """
    # Aggregate reservation counts per vehicle
    res_counts = (
        reservations.groupby("vehicle_id").size().reset_index(name="num_reservations")  # ty: ignore[no-matching-overload]
    )

    df = vehicles.merge(res_counts, on="vehicle_id", how="left")

    # CSV vehicles: observed zero is a real data point → fill with 0
    # UI vehicles: no observation yet → keep as NaN (will become NULL)
    csv_mask = df["source"] == "csv"
    df.loc[csv_mask, "num_reservations"] = (
        df.loc[csv_mask, "num_reservations"].fillna(0).astype(int)
    )
    # UI vehicles stay NaN — pd.Int64Dtype() allows nullable integers
    df["num_reservations"] = df["num_reservations"].astype("Int64")

    # Derived feature — the ONLY place this is computed
    df["price_diff"] = df["actual_price"] - df["recommended_price"]

    # Drop raw prices — not model features (price_diff captures the signal)
    df = df.drop(columns=["actual_price", "recommended_price"])

    # Feast requires a timestamp column
    df["event_timestamp"] = pd.Timestamp(datetime.now(tz=UTC))

    # Drop source column — not a feature
    df = df.drop(columns=["source"])

    logger.info("Computed features for %d vehicles.", len(df))
    return df


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write the feature DataFrame to Parquet for Feast's offline store."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def apply_and_materialize(feast_repo: str, features_df: pd.DataFrame) -> None:
    """Apply the Feast repo and materialize new arrivals to the online store.

    Vehicles with num_reservations = NULL (no observation yet) are written to
    the online store (Redis) for real-time inference. Vehicles with an observed
    count (including 0) already have outcomes and don't need predictions.
    """
    store = FeatureStore(repo_path=feast_repo)
    store.apply(
        [vehicle, vehicle_features_source, vehicle_features_view]  # type: ignore[list-item]
    )
    logger.info("Feast repo applied.")

    # Materialize only vehicles without observed outcomes to the online store
    new_arrivals = features_df[features_df["num_reservations"].isna()]
    if new_arrivals.empty:
        logger.info("No new arrivals to materialize to online store.")
        return

    store.write_to_online_store("vehicle_features", new_arrivals)
    logger.info("Materialized %d new arrivals to online store.", len(new_arrivals))


def run(db_path: str, feast_repo: str, parquet_path: str) -> None:
    """Full feature pipeline: load → compute → write → apply → materialize.

    The offline store (Parquet) only contains fleet vehicles — those with an
    observed num_reservations (including 0). This is the training dataset.

    New arrivals (num_reservations is NULL) are written only to the online
    store (Redis) for real-time inference.
    """
    vehicles, reservations = load_from_db(db_path)
    df = compute_features(vehicles, reservations)

    # Offline store: fleet vehicles only (observed outcomes for training)
    fleet = df[df["num_reservations"].notna()]
    write_parquet(fleet, parquet_path)

    # Online store: new arrivals only (no observed outcome, needs inference)
    apply_and_materialize(feast_repo, df)
    logger.info("Feature pipeline complete.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the feature pipeline."""
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
