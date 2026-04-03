"""
Vroom Forecast — Prediction Serving API

FastAPI application and route definitions. Model loading, feature engineering,
and schemas live in their own modules.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serving.config import settings
from serving.features import FEATURE_COLS, engineer_features
from serving.model import get_model, get_model_version, load_champion, predict
from serving.schemas import (
    BatchPredictionResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    PredictionResponse,
    VehicleFeatures,
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the champion model on startup."""
    load_champion()
    yield


app = FastAPI(
    title="Vroom Forecast API",
    description="Predicts the number of reservations for a vehicle based on its attributes.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check with model info."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=settings.model_name,
        model_version=get_model_version(),
        mlflow_uri=settings.mlflow_uri,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(vehicle: VehicleFeatures) -> PredictionResponse:
    """Predict reservation count for a single vehicle."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = predict([vehicle])
    return PredictionResponse(
        predicted_reservations=round(predictions[0], 2),
        model_version=get_model_version(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(vehicles: list[VehicleFeatures]) -> BatchPredictionResponse:
    """Predict reservation counts for multiple vehicles."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(vehicles) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 vehicles per batch")

    predictions = predict(vehicles)
    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                predicted_reservations=round(p, 2),
                model_version=get_model_version(),
            )
            for p in predictions
        ]
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Run N predictions and report latency statistics."""
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([req.vehicle.model_dump()])
    df = engineer_features(df)
    features = df[FEATURE_COLS]

    latencies: list[float] = []
    for _ in range(req.n_iterations):
        start = time.perf_counter()
        model.predict(features)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    arr = np.array(latencies)
    return BenchmarkResponse(
        n_iterations=req.n_iterations,
        avg_latency_ms=round(float(np.mean(arr)), 3),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
        model_version=get_model_version(),
    )
