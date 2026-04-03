"""
Vroom Forecast — Prediction Serving API

Serves reservation count predictions from the champion model in MLflow.
The model is loaded once at startup and cached in memory.

Endpoints:
    GET  /health           — liveness check + model info
    POST /predict           — predict reservation count for one vehicle
    POST /predict/batch     — predict for multiple vehicles
    POST /benchmark         — run N predictions and report latency stats
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Feature engineering (mirrors training/train.py) ──────────────────────────

FEATURE_COLS = [
    "technology",
    "actual_price",
    "recommended_price",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
    "price_ratio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived pricing features — must match training logic."""
    df = df.copy()
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"]
    return df


# ── Pydantic schemas ─────────────────────────────────────────────────────────


class VehicleFeatures(BaseModel):
    """Raw vehicle attributes (before feature engineering)."""

    technology: int = Field(..., ge=0, le=1, description="0=none, 1=installed")
    actual_price: float = Field(..., gt=0, description="Daily price set by owner")
    recommended_price: float = Field(..., gt=0, description="Market price")
    num_images: int = Field(..., ge=0, description="Number of photos")
    street_parked: int = Field(..., ge=0, le=1, description="0=no, 1=yes")
    description: int = Field(..., ge=0, description="Character count of description")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "technology": 1,
                    "actual_price": 45.0,
                    "recommended_price": 50.0,
                    "num_images": 8,
                    "street_parked": 0,
                    "description": 250,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_reservations: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class BenchmarkRequest(BaseModel):
    n_iterations: int = Field(default=1000, ge=1, le=100_000)
    vehicle: VehicleFeatures


class BenchmarkResponse(BaseModel):
    n_iterations: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    mlflow_uri: str


# ── Model loading ────────────────────────────────────────────────────────────

_model: Any = None
_model_version: str = ""


def load_champion_model() -> tuple[Any, str]:
    """Load the champion model from MLflow registry.

    Artifacts are shared via a volume mount (/mlartifacts), so the model
    can be loaded directly from disk after resolving the URI through
    the tracking server.
    """
    mlflow.set_tracking_uri(settings.mlflow_uri)
    client = mlflow.MlflowClient()

    champion_mv = client.get_model_version_by_alias(settings.model_name, "champion")
    version = champion_mv.version
    model_uri = champion_mv.source

    logger.info("Loading champion model: %s v%s from %s", settings.model_name, version, model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model, version


# ── Inference ────────────────────────────────────────────────────────────────


def predict_single(vehicle: VehicleFeatures) -> float:
    """Run inference for a single vehicle."""
    df = pd.DataFrame([vehicle.model_dump()])
    df = engineer_features(df)
    prediction = _model.predict(df[FEATURE_COLS])
    return float(prediction[0])


# ── App lifecycle ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the model on startup."""
    global _model, _model_version  # noqa: PLW0603
    _model, _model_version = load_champion_model()
    yield


app = FastAPI(
    title="Vroom Forecast API",
    description="Predicts the number of reservations for a vehicle based on its attributes.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check with model info."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=settings.model_name,
        model_version=_model_version,
        mlflow_uri=settings.mlflow_uri,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(vehicle: VehicleFeatures) -> PredictionResponse:
    """Predict reservation count for a single vehicle."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prediction = predict_single(vehicle)
    return PredictionResponse(
        predicted_reservations=round(prediction, 2),
        model_version=_model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(vehicles: list[VehicleFeatures]) -> BatchPredictionResponse:
    """Predict reservation counts for multiple vehicles."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(vehicles) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 vehicles per batch")

    df = pd.DataFrame([v.model_dump() for v in vehicles])
    df = engineer_features(df)
    predictions = _model.predict(df[FEATURE_COLS])

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                predicted_reservations=round(float(p), 2),
                model_version=_model_version,
            )
            for p in predictions
        ]
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Run N predictions and report latency statistics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([req.vehicle.model_dump()])
    df = engineer_features(df)
    features = df[FEATURE_COLS]

    latencies: list[float] = []
    for _ in range(req.n_iterations):
        start = time.perf_counter()
        _model.predict(features)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    arr = np.array(latencies)
    return BenchmarkResponse(
        n_iterations=req.n_iterations,
        avg_latency_ms=round(float(np.mean(arr)), 3),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
        model_version=_model_version,
    )
