"""Champion model loading, feature store access, inference, and model reload via Redis pub/sub."""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlflow
import pandas as pd

from serving.config import settings
from serving.features import FEATURE_COLS, engineer_features
from serving.schemas import VehicleFeatures

logger = logging.getLogger(__name__)

_model: Any = None
_model_version: str = ""
_feast_store: Any = None
_reload_thread: threading.Thread | None = None

REDIS_CHANNEL = "vroom-forecast:model-promoted"

# Feature references for the online store lookup (must match definitions.py)
_ONLINE_FEATURE_REFS = [
    "vehicle_features:technology",
    "vehicle_features:actual_price",
    "vehicle_features:recommended_price",
    "vehicle_features:num_images",
    "vehicle_features:street_parked",
    "vehicle_features:description",
    "vehicle_features:price_diff",
    "vehicle_features:price_ratio",
]


# ── Model loading ────────────────────────────────────────────────────────────


def load_champion() -> None:
    """Load the champion model from MLflow into module-level state.

    If no champion model exists yet, logs a warning and leaves the model
    unloaded. The /health endpoint will return 503 until a model is available.
    """
    global _model, _model_version  # noqa: PLW0603

    mlflow.set_tracking_uri(settings.mlflow_uri)
    client = mlflow.MlflowClient()

    try:
        champion_mv = client.get_model_version_by_alias(settings.model_name, "champion")
    except Exception:
        logger.warning(
            "No champion model found for '%s'. Run the training + promotion pipeline first.",
            settings.model_name,
        )
        return

    new_version = champion_mv.version

    if new_version == _model_version and _model is not None:
        logger.info("Champion v%s already loaded — skipping reload.", new_version)
        return

    model_uri = champion_mv.source
    logger.info(
        "Loading champion model: %s v%s from %s",
        settings.model_name,
        new_version,
        model_uri,
    )
    _model = mlflow.sklearn.load_model(model_uri)
    _model_version = new_version
    logger.info("Model loaded successfully.")


# ── Redis pub/sub listener ───────────────────────────────────────────────────


def _subscribe_loop() -> None:
    """Background thread: subscribe to Redis and reload model on promotion events."""
    import redis

    r = redis.from_url(settings.redis_url)
    pubsub = r.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)
    logger.info("Subscribed to Redis channel '%s' for model reload events.", REDIS_CHANNEL)

    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        logger.info("Received promotion event: %s", message["data"])
        try:
            load_champion()
        except Exception:
            logger.exception("Failed to reload champion model after promotion event.")


def start_reload_listener() -> None:
    """Start the background Redis subscriber thread (if Redis is configured)."""
    global _reload_thread  # noqa: PLW0603

    if settings.redis_url is None:
        logger.info("No SERVING_REDIS_URL configured — model reload listener disabled.")
        return

    _reload_thread = threading.Thread(target=_subscribe_loop, daemon=True, name="model-reload")
    _reload_thread.start()
    logger.info("Model reload listener started.")


# ── Feast online store ───────────────────────────────────────────────────────


def init_feast() -> None:
    """Initialize the Feast online store client (if configured and available)."""
    global _feast_store  # noqa: PLW0603

    if settings.feast_repo is None:
        logger.info("No SERVING_FEAST_REPO configured — online store disabled.")
        return

    from pathlib import Path

    repo_path = Path(settings.feast_repo)
    if not (repo_path / "feature_store.yaml").exists():
        logger.warning(
            "Feast repo not found at %s — online store disabled. "
            "Run the materialize pipeline to populate it.",
            settings.feast_repo,
        )
        return

    from feast import FeatureStore

    _feast_store = FeatureStore(repo_path=settings.feast_repo)
    logger.info("Feast online store initialized from %s.", settings.feast_repo)


# ── Accessors ────────────────────────────────────────────────────────────────


def get_model() -> Any:
    """Return the loaded model, or None if not yet loaded."""
    return _model


def get_model_version() -> str:
    """Return the loaded model's version string."""
    return _model_version


def feast_available() -> bool:
    """Return True if the Feast online store is configured and initialized."""
    return _feast_store is not None


# ── Feature lookup ───────────────────────────────────────────────────────────


def get_online_features(vehicle_ids: list[int]) -> pd.DataFrame:
    """Look up pre-computed features from the Feast online store (Redis)."""
    if _feast_store is None:
        raise RuntimeError("Feast online store is not initialized.")

    entity_rows = [{"vehicle_id": vid} for vid in vehicle_ids]
    response = _feast_store.get_online_features(
        features=_ONLINE_FEATURE_REFS,
        entity_rows=entity_rows,
    )
    return response.to_df()


# ── Inference ────────────────────────────────────────────────────────────────


def predict_from_features(vehicles: list[VehicleFeatures]) -> list[float]:
    """Run inference from raw vehicle attributes (computes features on the fly)."""
    df = pd.DataFrame([v.model_dump() for v in vehicles])
    df = engineer_features(df)
    raw = _model.predict(df[FEATURE_COLS])
    return [float(p) for p in raw]


def predict_from_ids(vehicle_ids: list[int]) -> list[float]:
    """Run inference by looking up pre-computed features from the Feast online store (Redis).

    Features must have been materialized by the feature pipeline.
    User-saved vehicles are available after the next materialization run.
    """
    if _feast_store is None:
        raise RuntimeError("Feast online store is not initialized.")

    df = get_online_features(vehicle_ids)
    raw = _model.predict(df[FEATURE_COLS])
    return [float(p) for p in raw]
