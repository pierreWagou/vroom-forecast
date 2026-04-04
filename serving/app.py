"""
Vroom Forecast — Ray Serve Application

FastAPI app with Ray Serve deployments composed into a single application.
The ingress (FastAPI) holds handles to the Predictor, FeatureComputer,
and FeatureLookup deployments.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
from ray.serve.handle import DeploymentHandle

from serving.config import settings
from serving.schemas import (
    BatchPredictionResponse,
    BenchmarkByIdRequest,
    BenchmarkRequest,
    BenchmarkResponse,
    ComputedFeatures,
    HealthResponse,
    PredictionResponse,
    ReloadResponse,
    SaveVehicleResponse,
    VehicleFeatures,
    VehicleIdRequest,
    VehicleRecord,
)
from serving.vehicles import list_vehicles, save_vehicle

app = FastAPI(
    title="Vroom Forecast API",
    description="Predicts vehicle reservations — powered by Ray Serve.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@serve.deployment
@serve.ingress(app)
class VroomForecastApp:
    """Ray Serve ingress — holds handles to all downstream deployments."""

    def __init__(
        self,
        predictor: DeploymentHandle,
        feature_computer: DeploymentHandle,
        feature_lookup: DeploymentHandle,
    ) -> None:
        self.predictor = predictor
        self.feature_computer = feature_computer
        self.feature_lookup = feature_lookup

    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        version = await self.predictor.get_version.remote()
        feast_online = await self.feature_lookup.is_available.remote()
        return HealthResponse(
            status="ok",
            model_name=settings.model_name,
            model_version=version,
            mlflow_uri=settings.mlflow_uri,
            feast_online=feast_online,
        )

    @app.post("/reload", response_model=ReloadResponse)
    async def reload_model(self) -> ReloadResponse:
        previous, current = await self.predictor.reload.remote()
        return ReloadResponse(
            status="reloaded" if current != previous else "unchanged",
            previous_version=previous,
            current_version=current,
        )

    @app.post("/vehicles", response_model=SaveVehicleResponse)
    async def save_vehicle_endpoint(self, vehicle: VehicleFeatures) -> SaveVehicleResponse:
        import asyncio

        vehicle_id = await asyncio.to_thread(save_vehicle, vehicle)
        return SaveVehicleResponse(vehicle_id=vehicle_id, status="saved")

    @app.get("/vehicles", response_model=list[VehicleRecord])
    async def list_vehicles_endpoint(self) -> list[VehicleRecord]:
        import asyncio

        return await asyncio.to_thread(list_vehicles)

    @app.get("/vehicles/{vehicle_id}/features", response_model=ComputedFeatures)
    async def get_vehicle_features(self, vehicle_id: int) -> ComputedFeatures:
        """Get the computed features from the online store for a saved vehicle."""
        feast_online = await self.feature_lookup.is_available.remote()
        if not feast_online:
            return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)
        try:
            df = await self.feature_lookup.lookup.remote([vehicle_id])
            row = df.iloc[0]
            # Check if Feast returned nulls (vehicle not materialized)
            if pd.isna(row.get("technology")):
                return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)
            return ComputedFeatures(
                vehicle_id=vehicle_id,
                technology=int(row["technology"]),
                actual_price=float(row["actual_price"]),
                recommended_price=float(row["recommended_price"]),
                num_images=int(row["num_images"]),
                street_parked=int(row["street_parked"]),
                description=int(row["description"]),
                price_diff=float(row["price_diff"]),
                price_ratio=float(row["price_ratio"]),
                materialized=True,
            )
        except Exception:
            return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(self, vehicle: VehicleFeatures) -> PredictionResponse:
        features_df = await self.feature_computer.compute.remote([vehicle])
        predictions = await self.predictor.predict.remote(features_df)
        version = await self.predictor.get_version.remote()
        return PredictionResponse(
            predicted_reservations=round(predictions[0], 2),
            model_version=version,
        )

    @app.post("/predict/id", response_model=PredictionResponse)
    async def predict_by_id(self, req: VehicleIdRequest) -> PredictionResponse:
        feast_online = await self.feature_lookup.is_available.remote()
        if not feast_online:
            raise HTTPException(status_code=503, detail="Feature store not configured")
        features_df = await self.feature_lookup.lookup.remote([req.vehicle_id])
        if features_df.isna().any().any():
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle {req.vehicle_id} not found in the online store",
            )
        predictions = await self.predictor.predict.remote(features_df)
        version = await self.predictor.get_version.remote()
        return PredictionResponse(
            predicted_reservations=round(predictions[0], 2),
            model_version=version,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(self, vehicles: list[VehicleFeatures]) -> BatchPredictionResponse:
        if len(vehicles) > 1000:
            raise HTTPException(status_code=400, detail="Max 1000 vehicles per batch")
        if len(vehicles) == 0:
            return BatchPredictionResponse(predictions=[])
        features_df = await self.feature_computer.compute.remote(vehicles)
        predictions = await self.predictor.predict.remote(features_df)
        version = await self.predictor.get_version.remote()
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    predicted_reservations=round(p, 2),
                    model_version=version,
                )
                for p in predictions
            ]
        )

    @app.post("/benchmark", response_model=BenchmarkResponse)
    async def benchmark(self, req: BenchmarkRequest) -> BenchmarkResponse:
        """Benchmark: feature computation (on the fly) + inference."""
        latencies: list[float] = []
        feature_times: list[float] = []
        predict_times: list[float] = []
        for _ in range(req.n_iterations):
            total_start = time.perf_counter()

            feat_start = time.perf_counter()
            features_df = await self.feature_computer.compute.remote([req.vehicle])
            feat_elapsed = (time.perf_counter() - feat_start) * 1000
            feature_times.append(feat_elapsed)

            pred_start = time.perf_counter()
            await self.predictor.predict.remote(features_df)
            pred_elapsed = (time.perf_counter() - pred_start) * 1000
            predict_times.append(pred_elapsed)

            total_elapsed = (time.perf_counter() - total_start) * 1000
            latencies.append(total_elapsed)

        version = await self.predictor.get_version.remote()
        arr = np.array(latencies)
        return BenchmarkResponse(
            n_iterations=req.n_iterations,
            avg_latency_ms=round(float(np.mean(arr)), 3),
            p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
            p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
            model_version=version,
            source="raw",
            avg_features_ms=round(float(np.mean(feature_times)), 3),
            avg_predict_ms=round(float(np.mean(predict_times)), 3),
        )

    @app.post("/benchmark/id", response_model=BenchmarkResponse)
    async def benchmark_by_id(self, req: BenchmarkByIdRequest) -> BenchmarkResponse:
        """Benchmark: feature lookup (online store) + inference."""
        feast_online = await self.feature_lookup.is_available.remote()
        if not feast_online:
            raise HTTPException(status_code=503, detail="Feature store not configured")
        latencies: list[float] = []
        feature_times: list[float] = []
        predict_times: list[float] = []
        for _ in range(req.n_iterations):
            total_start = time.perf_counter()

            feat_start = time.perf_counter()
            features_df = await self.feature_lookup.lookup.remote([req.vehicle_id])
            feat_elapsed = (time.perf_counter() - feat_start) * 1000
            feature_times.append(feat_elapsed)

            pred_start = time.perf_counter()
            await self.predictor.predict.remote(features_df)
            pred_elapsed = (time.perf_counter() - pred_start) * 1000
            predict_times.append(pred_elapsed)

            total_elapsed = (time.perf_counter() - total_start) * 1000
            latencies.append(total_elapsed)

        version = await self.predictor.get_version.remote()
        arr = np.array(latencies)
        return BenchmarkResponse(
            n_iterations=req.n_iterations,
            avg_latency_ms=round(float(np.mean(arr)), 3),
            p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
            p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
            model_version=version,
            source="online_store",
            avg_features_ms=round(float(np.mean(feature_times)), 3),
            avg_predict_ms=round(float(np.mean(predict_times)), 3),
        )
