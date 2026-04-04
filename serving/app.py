"""
Vroom Forecast — Prediction Serving API

FastAPI application and route definitions. Model loading, feature engineering,
and schemas live in their own modules.

Supports two prediction modes:
  - POST /predict — raw vehicle attributes (features computed on the fly)
  - POST /predict/id — vehicle ID lookup from the Feast online store (Redis)
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
from serving.model import (
    feast_available,
    get_model,
    get_model_version,
    init_feast,
    load_champion,
    predict_from_features,
    predict_from_ids,
    start_reload_listener,
)
from serving.schemas import (
    BatchPredictionResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    PredictionResponse,
    ReloadResponse,
    SaveVehicleResponse,
    VehicleFeatures,
    VehicleIdRequest,
    VehicleRecord,
)
from serving.vehicles import list_vehicles, save_vehicle


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the champion model, initialize feature store, and start reload listener."""
    load_champion()
    init_feast()
    start_reload_listener()
    yield


app = FastAPI(
    title="Vroom Forecast API",
    description="Predicts the number of reservations for a vehicle based on its attributes.",
    version="0.2.0",
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
    """Liveness check with model info and feature store status."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=settings.model_name,
        model_version=get_model_version(),
        mlflow_uri=settings.mlflow_uri,
        feast_online=feast_available(),
    )


@app.post("/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    """Reload the champion model from MLflow."""
    previous = get_model_version()
    load_champion()
    current = get_model_version()
    return ReloadResponse(
        status="reloaded" if current != previous else "unchanged",
        previous_version=previous,
        current_version=current,
    )


# ── Vehicle management ───────────────────────────────────────────────────────


@app.post("/vehicles", response_model=SaveVehicleResponse)
def save_vehicle_endpoint(vehicle: VehicleFeatures) -> SaveVehicleResponse:
    """Save a vehicle to the database and push its features to Redis.

    The vehicle gets an auto-incremented ID and can be used with /predict/id.
    Features are immediately available in Redis — no materialization needed.
    """
    vehicle_id = save_vehicle(vehicle)
    return SaveVehicleResponse(vehicle_id=vehicle_id, status="saved")


@app.get("/vehicles", response_model=list[VehicleRecord])
def list_vehicles_endpoint() -> list[VehicleRecord]:
    """List all saved vehicles."""
    return list_vehicles()


# ── Prediction ───────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
def predict_single(vehicle: VehicleFeatures) -> PredictionResponse:
    """Predict reservation count from raw vehicle attributes (features computed on the fly)."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = predict_from_features([vehicle])
    return PredictionResponse(
        predicted_reservations=round(predictions[0], 2),
        model_version=get_model_version(),
    )


@app.post("/predict/id", response_model=PredictionResponse)
def predict_by_id(req: VehicleIdRequest) -> PredictionResponse:
    """Predict reservation count by vehicle ID (features from the online store)."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not feast_available():
        raise HTTPException(status_code=503, detail="Feature store not configured")
    predictions = predict_from_ids([req.vehicle_id])
    return PredictionResponse(
        predicted_reservations=round(predictions[0], 2),
        model_version=get_model_version(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(vehicles: list[VehicleFeatures]) -> BatchPredictionResponse:
    """Predict reservation counts for multiple vehicles (features computed on the fly)."""
    if get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(vehicles) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 vehicles per batch")
    if len(vehicles) == 0:
        return BatchPredictionResponse(predictions=[])

    predictions = predict_from_features(vehicles)
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
