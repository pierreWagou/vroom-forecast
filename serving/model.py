"""Ray Serve deployments and actors for model inference and feature computation."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd
import ray
from ray import serve

from serving.config import settings
from serving.features import FEATURE_COLS, engineer_features
from serving.schemas import VehicleFeatures

logger = logging.getLogger(__name__)

# Redis pub/sub channels — single source of truth for the serving layer
VEHICLE_SAVED_CHANNEL = "vroom-forecast:vehicle-saved"
VEHICLE_MATERIALIZED_CHANNEL = "vroom-forecast:vehicle-materialized"
MODEL_PROMOTED_CHANNEL = "vroom-forecast:model-promoted"

# Feature references for Feast online store lookup
_ONLINE_FEATURE_REFS = [
    "vehicle_features:technology",
    "vehicle_features:num_images",
    "vehicle_features:street_parked",
    "vehicle_features:description",
    "vehicle_features:price_diff",
]


@serve.deployment(max_ongoing_requests=10)
class Predictor:
    """Loads the champion model from MLflow and runs inference.

    Model is loaded in __init__ (deployment startup). To reload after
    promotion, redeploy this deployment — Ray Serve handles zero-downtime
    rolling updates.
    """

    def __init__(self) -> None:
        """Load the champion model from MLflow on deployment startup."""
        self.model: Any = None
        self.model_version: str = ""
        self._load_champion()

    def _load_champion(self) -> None:
        """Fetch and load the champion model version from MLflow."""
        mlflow.set_tracking_uri(settings.mlflow_uri)
        client = mlflow.MlflowClient()

        try:
            champion_mv = client.get_model_version_by_alias(settings.model_name, "champion")
        except mlflow.exceptions.MlflowException:
            logger.warning(
                "No champion model found for '%s'. Predictor started without a model.",
                settings.model_name,
            )
            return
        except Exception:
            logger.warning(
                "Could not reach MLflow at '%s'. Predictor started without a model.",
                settings.mlflow_uri,
                exc_info=True,
            )
            return

        self.model_version = champion_mv.version
        model_uri = champion_mv.source
        logger.info("Loading champion model: %s v%s", settings.model_name, self.model_version)
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully.")

    def predict(self, features_df: pd.DataFrame) -> list[float]:
        """Run inference on a DataFrame of features."""
        if self.model is None:
            raise RuntimeError("No model loaded.")
        raw = self.model.predict(features_df[FEATURE_COLS])
        return [float(p) for p in raw]

    def benchmark_predict(self, features_df: pd.DataFrame, n: int) -> list[float]:
        """Run predict N times and return per-iteration latencies in ms."""
        import time

        if self.model is None:
            raise RuntimeError("No model loaded.")
        cols = features_df[FEATURE_COLS]
        latencies: list[float] = []
        for _ in range(n):
            start = time.perf_counter()
            self.model.predict(cols)
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies

    def reload(self) -> tuple[str, str]:
        """Reload the champion model. Returns (previous_version, current_version)."""
        previous = self.model_version
        self._load_champion()
        return previous, self.model_version

    def get_version(self) -> str:
        """Return the currently loaded model version string."""
        return self.model_version

    def is_loaded(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self.model is not None


@serve.deployment(max_ongoing_requests=10)
class FeatureComputer:
    """Computes derived features from raw vehicle attributes.

    Stateless — no initialization needed. This is the serving-side
    feature computation, equivalent to the feature pipeline's logic.
    """

    def compute(self, vehicles: list[VehicleFeatures]) -> pd.DataFrame:
        """Compute features from raw attributes."""
        df = pd.DataFrame([v.model_dump() for v in vehicles])
        return engineer_features(df)

    def benchmark_compute(self, vehicles: list[VehicleFeatures], n: int) -> list[float]:
        """Run compute N times and return per-iteration latencies in ms."""
        import time

        latencies: list[float] = []
        for _ in range(n):
            start = time.perf_counter()
            self.compute(vehicles)
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies


@serve.deployment(max_ongoing_requests=10)
class FeatureLookup:
    """Looks up pre-computed features from the Feast online store (Redis).

    Initialized with the Feast repo path. If the repo doesn't exist,
    the deployment starts but lookup calls will raise.
    """

    def __init__(self) -> None:
        """Initialize the Feast online store connection."""
        self._store: Any = None
        self._init_feast()

    def _init_feast(self) -> None:
        """Connect to the Feast feature store using the configured repo path."""
        if settings.feast_repo is None:
            logger.info("No SERVING_FEAST_REPO configured — online store disabled.")
            return

        from pathlib import Path

        repo_path = Path(settings.feast_repo)
        if not (repo_path / "feature_store.yaml").exists():
            logger.warning(
                "Feast repo not found at %s — online store disabled.", settings.feast_repo
            )
            return

        from feast import FeatureStore

        self._store = FeatureStore(repo_path=settings.feast_repo)
        logger.info("Feast online store initialized from %s.", settings.feast_repo)

    def lookup(self, vehicle_ids: list[int]) -> pd.DataFrame:
        """Look up features from Feast online store."""
        if self._store is None:
            raise RuntimeError("Feast online store is not initialized.")
        entity_rows = [{"vehicle_id": vid} for vid in vehicle_ids]
        response = self._store.get_online_features(
            features=_ONLINE_FEATURE_REFS,
            entity_rows=entity_rows,
        )
        return response.to_df()

    def benchmark_lookup(self, vehicle_ids: list[int], n: int) -> list[float]:
        """Run lookup N times and return per-iteration latencies in ms."""
        import time

        if self._store is None:
            raise RuntimeError("Feast online store is not initialized.")
        entity_rows = [{"vehicle_id": vid} for vid in vehicle_ids]
        latencies: list[float] = []
        for _ in range(n):
            start = time.perf_counter()
            self._store.get_online_features(
                features=_ONLINE_FEATURE_REFS,
                entity_rows=entity_rows,
            )
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies

    def is_available(self) -> bool:
        """Return True if the Feast online store is connected."""
        return self._store is not None

    def get_feature_view_info(self) -> dict | None:
        """Read the feature view definition from the Feast registry."""
        if self._store is None:
            return None
        try:
            fv = self._store.get_feature_view("vehicle_features")
            entity = fv.entities[0] if fv.entities else "unknown"
            schema_fields = [f.name for f in fv.schema]
            # Separate features from label (num_reservations is the label)
            label = "num_reservations"
            features = [f for f in schema_fields if f != label]
            ttl_days = fv.ttl.days if fv.ttl else None
            return {
                "name": fv.name,
                "entity": entity,
                "entity_key": "vehicle_id",
                "features": features,
                "label": label,
                "ttl_days": ttl_days,
            }
        except Exception:
            return None


@serve.deployment
class OfflineFeatureReader:
    """Reads pre-computed features from the Feast offline store (Parquet).

    Used for fleet vehicles (source=csv) whose features were computed by the
    batch pipeline. The Parquet file is the single source of truth for the
    offline store — the same data used for model training.
    """

    def __init__(self) -> None:
        """Load the offline store Parquet file into memory."""
        self._df: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        """Read the Parquet file from the configured path and index by vehicle_id."""
        from pathlib import Path

        path = settings.offline_store_path
        if path is None:
            logger.info("No SERVING_OFFLINE_STORE_PATH configured — offline reader disabled.")
            return

        parquet_path = Path(path)
        if not parquet_path.exists():
            logger.warning(
                "Offline store not found at %s — disabled. Run the materialize pipeline first.",
                path,
            )
            return

        self._df = pd.read_parquet(parquet_path)
        self._df = self._df.set_index("vehicle_id")
        logger.info("OfflineFeatureReader: Loaded %d vehicles from %s.", len(self._df), path)

    def lookup(self, vehicle_ids: list[int]) -> pd.DataFrame | None:
        """Look up features for the given vehicle IDs from the Parquet file.

        Returns None if the offline store is not available, or a DataFrame
        with only the vehicles found.
        """
        if self._df is None:
            return None
        mask = self._df.index.isin(vehicle_ids)
        if not mask.any():
            return None
        return self._df.loc[mask].reset_index()

    def is_available(self) -> bool:
        """Return True if the offline store Parquet file is loaded."""
        return self._df is not None


@ray.remote
class ModelReloadListener:
    """Ray actor that subscribes to Redis pub/sub and reloads the Predictor
    when a model promotion event is received.

    This bridges the promotion pipeline → serving: promotion publishes to
    `model-promoted` channel, this actor calls Predictor.reload().
    """

    def __init__(self, predictor_handle: Any) -> None:
        """Store the Predictor handle for reload calls."""
        self._predictor = predictor_handle

    def run(self) -> None:
        """Subscribe to Redis and reload the model on promotion events."""
        if settings.redis_url is None:
            logger.info("ModelReloadListener: No Redis URL — exiting.")
            return

        import redis

        r = redis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        pubsub.subscribe(MODEL_PROMOTED_CHANNEL)
        logger.info("ModelReloadListener: Subscribed to '%s'.", MODEL_PROMOTED_CHANNEL)

        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                logger.info("ModelReloadListener: Got promotion event: %s", message["data"])
                ray.get(self._predictor.reload.remote())
                logger.info("ModelReloadListener: Model reloaded.")
            except Exception as e:
                logger.error("ModelReloadListener: Reload failed: %s", e)


@ray.remote
class FeatureMaterializer:
    """Ray actor that subscribes to Redis pub/sub and materializes features
    to the Feast online store in real time.

    This is a long-running actor (not a Serve deployment) — it doesn't
    handle HTTP. It runs inside the same Ray cluster as the Serve deployments.
    """

    def __init__(self) -> None:
        """Initialize the Feast store for writing to the online store."""
        self._store: Any = None
        self._redis_url = settings.redis_url
        self._init_feast()

    def _init_feast(self) -> None:
        """Set up the Feast feature store and apply entity/view definitions."""
        if settings.feast_repo is None:
            logger.info("FeatureMaterializer: no Feast repo configured — disabled.")
            return

        import sys
        from pathlib import Path

        repo_path = Path(settings.feast_repo)
        if not (repo_path / "feature_store.yaml").exists():
            logger.warning(
                "FeatureMaterializer: Feast repo not found at %s — disabled.",
                settings.feast_repo,
            )
            return

        # Add the feast repo parent to sys.path so definitions.py can be imported
        sys.path.insert(0, str(repo_path.parent))

        from feast import FeatureStore
        from feature_repo.definitions import (
            vehicle,
            vehicle_features_source,
            vehicle_features_view,
        )

        self._store = FeatureStore(repo_path=settings.feast_repo)
        self._store.apply(
            [vehicle, vehicle_features_source, vehicle_features_view]  # type: ignore[list-item]
        )
        logger.info("FeatureMaterializer: Feast store initialized.")

    def _compute_and_write(self, vehicle_id: int, raw: dict) -> None:
        """Compute derived features and write to the Feast online store."""
        if self._store is None:
            logger.warning(
                "FeatureMaterializer: store not initialized, skipping vehicle %d", vehicle_id
            )
            return

        # Reuse the shared engineer_features function to avoid formula duplication
        df = pd.DataFrame([{**raw, "vehicle_id": vehicle_id}])
        df = engineer_features(df)

        row = {
            "vehicle_id": vehicle_id,
            "technology": df.iloc[0]["technology"],
            "num_images": df.iloc[0]["num_images"],
            "street_parked": df.iloc[0]["street_parked"],
            "description": df.iloc[0]["description"],
            "price_diff": df.iloc[0]["price_diff"],
            "num_reservations": pd.NA,
            "event_timestamp": pd.Timestamp(datetime.now(tz=UTC)),
        }

        write_df = pd.DataFrame([row])
        self._store.write_to_online_store("vehicle_features", write_df)
        logger.info("FeatureMaterializer: Materialized vehicle #%d", vehicle_id)

        # Notify listeners (UI SSE) that this vehicle is now materialized
        if self._redis_url:
            import redis as redis_lib

            try:
                r = redis_lib.from_url(self._redis_url)
                try:
                    r.publish(
                        VEHICLE_MATERIALIZED_CHANNEL,
                        json.dumps({"vehicle_id": vehicle_id}),
                    )
                finally:
                    r.close()
            except Exception:
                logger.warning("Failed to publish materialized event for vehicle %d", vehicle_id)

    def run(self) -> None:
        """Subscribe to Redis and process vehicle-saved events forever."""
        if settings.redis_url is None:
            logger.info("FeatureMaterializer: No Redis URL — exiting.")
            return

        import redis

        r = redis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        pubsub.subscribe(VEHICLE_SAVED_CHANNEL)
        logger.info("FeatureMaterializer: Subscribed to '%s'.", VEHICLE_SAVED_CHANNEL)

        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                logger.info("FeatureMaterializer: Got event: %s", message["data"])
                raw = json.loads(message["data"])
                vehicle_id = raw.pop("vehicle_id")
                self._compute_and_write(vehicle_id, raw)
            except Exception as e:
                logger.error("FeatureMaterializer: Error: %s", e)
