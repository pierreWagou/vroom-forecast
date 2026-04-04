"""
Feature Worker — Real-time feature materialization via Redis pub/sub.

Subscribes to the "vroom-forecast:vehicle-saved" channel. When a new vehicle
is saved, computes its features and writes them to the Feast online store
(Redis), making them available for /predict/id immediately.

This complements the batch materialization pipeline (pipeline.py), which runs
daily for the full catalog. The worker handles the real-time path for newly
saved vehicles.

Usage:
    uv run python worker.py --feast-repo feature_repo --redis-url redis://redis:6379
"""

from __future__ import annotations

import argparse
import json
import logging

import redis as redis_lib
from feast import FeatureStore
from feature_repo.definitions import vehicle, vehicle_features_source, vehicle_features_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VEHICLE_SAVED_CHANNEL = "vroom-forecast:vehicle-saved"
VEHICLE_ID_OFFSET = 100_000


def compute_features(raw: dict) -> dict:
    """Compute derived features from raw vehicle attributes.

    This is the SAME logic as pipeline.py — single source of truth.
    """
    return {
        **raw,
        "price_diff": raw["actual_price"] - raw["recommended_price"],
        "price_ratio": raw["actual_price"] / raw["recommended_price"],
    }


def write_to_online_store(store: FeatureStore, vehicle_id: int, features: dict) -> None:
    """Write a single vehicle's features to the Feast online store."""
    from datetime import UTC, datetime

    import pandas as pd

    # Build a single-row DataFrame matching the FeatureView schema
    row = {
        "vehicle_id": vehicle_id,
        "technology": features["technology"],
        "actual_price": features["actual_price"],
        "recommended_price": features["recommended_price"],
        "num_images": features["num_images"],
        "street_parked": features["street_parked"],
        "description": features["description"],
        "price_diff": features["price_diff"],
        "price_ratio": features["price_ratio"],
        "num_reservations": 0,  # New vehicle, no reservations yet
        "event_timestamp": pd.Timestamp(datetime.now(tz=UTC)),
    }

    df = pd.DataFrame([row])
    store.write_to_online_store("vehicle_features", df)
    logger.info("Wrote features for vehicle %d to online store.", vehicle_id)


def run(feast_repo: str, redis_url: str) -> None:
    """Subscribe to vehicle-saved events and materialize features in real time."""
    # Initialize Feast
    store = FeatureStore(repo_path=feast_repo)
    store.apply(
        [vehicle, vehicle_features_source, vehicle_features_view]  # type: ignore[list-item]
    )
    logger.info("Feast store initialized from %s.", feast_repo)

    # Subscribe to Redis channel
    r = redis_lib.from_url(redis_url)
    pubsub = r.pubsub()
    pubsub.subscribe(VEHICLE_SAVED_CHANNEL)
    logger.info("Subscribed to '%s'. Waiting for events...", VEHICLE_SAVED_CHANNEL)

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            raw = json.loads(message["data"])
            db_id = raw.pop("vehicle_id")
            # Offset to match the feature pipeline convention
            feature_store_id = db_id + VEHICLE_ID_OFFSET

            features = compute_features(raw)
            write_to_online_store(store, feature_store_id, features)
            logger.info(
                "Materialized vehicle DB#%d → FS#%d in real time.",
                db_id,
                feature_store_id,
            )
        except Exception:
            logger.exception("Failed to process vehicle-saved event: %s", message["data"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time feature materialization worker")
    parser.add_argument(
        "--feast-repo",
        type=str,
        default="feature_repo",
        help="Path to the Feast feature repo directory",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis URL for pub/sub subscription",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(feast_repo=args.feast_repo, redis_url=args.redis_url)
