"""Champion model loading and inference."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import pandas as pd

from serving.config import settings
from serving.features import FEATURE_COLS, engineer_features
from serving.schemas import VehicleFeatures

logger = logging.getLogger(__name__)

_model: Any = None
_model_version: str = ""


def load_champion() -> None:
    """Load the champion model from MLflow into module-level state.

    Loads using the models:/<name>@champion URI, which resolves through
    the MLflow tracking server. Artifacts must be accessible to this process
    (via shared volume mount or artifact proxy).
    """
    global _model, _model_version  # noqa: PLW0603

    mlflow.set_tracking_uri(settings.mlflow_uri)
    client = mlflow.MlflowClient()

    champion_mv = client.get_model_version_by_alias(settings.model_name, "champion")
    _model_version = champion_mv.version

    # Use the model version's source URI directly (works for local artifact stores)
    model_uri = champion_mv.source

    logger.info(
        "Loading champion model: %s v%s from %s",
        settings.model_name,
        _model_version,
        model_uri,
    )
    _model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully.")


def get_model() -> Any:
    """Return the loaded model, or None if not yet loaded."""
    return _model


def get_model_version() -> str:
    """Return the loaded model's version string."""
    return _model_version


def predict(vehicles: list[VehicleFeatures]) -> list[float]:
    """Run inference for one or more vehicles. Returns list of predictions."""
    df = pd.DataFrame([v.model_dump() for v in vehicles])
    df = engineer_features(df)
    raw = _model.predict(df[FEATURE_COLS])
    return [float(p) for p in raw]
