"""
Vroom Forecast — Ray Serve Application

FastAPI app with Ray Serve deployments composed into a single application.
The ingress (FastAPI) holds handles to the Predictor, FeatureComputer,
and FeatureLookup deployments.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.responses import StreamingResponse

from serving.config import settings
from serving.schemas import (
    BatchPredictionResponse,
    BenchmarkByIdRequest,
    BenchmarkRequest,
    BenchmarkResponse,
    ComputedFeatures,
    DagRunStatus,
    DeleteVehicleResponse,
    FeatureViewInfo,
    HealthResponse,
    PredictionResponse,
    ReloadResponse,
    SaveVehicleResponse,
    StoreDetails,
    StoreInfoResponse,
    TriggerDagResponse,
    VehicleFeatures,
    VehicleIdRequest,
    VehicleRecord,
)
from serving.vehicles import delete_vehicle, list_vehicles, save_vehicle

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


@serve.deployment(max_ongoing_requests=100)
@serve.ingress(app)
class VroomForecastApp:
    """Ray Serve ingress — holds handles to all downstream deployments."""

    def __init__(
        self,
        predictor: DeploymentHandle,
        feature_computer: DeploymentHandle,
        feature_lookup: DeploymentHandle,
        offline_reader: DeploymentHandle,
    ) -> None:
        """Store handles to all downstream Ray Serve deployments."""
        self.predictor = predictor
        self.feature_computer = feature_computer
        self.feature_lookup = feature_lookup
        self.offline_reader = offline_reader

    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        """Check API health, model status, and feature store availability."""
        is_loaded = await self.predictor.is_loaded.remote()
        is_loading = await self.predictor.is_loading.remote()
        feast_online = await self.feature_lookup.is_available.remote()

        if is_loading:
            status = "loading"
        elif is_loaded:
            status = "ok"
        else:
            status = "no_model"

        version = ""
        if is_loaded:
            version = await self.predictor.get_version.remote()

        return HealthResponse(
            status=status,
            model_name=settings.model_name,
            model_version=version,
            mlflow_uri=settings.mlflow_uri,
            feast_online=feast_online,
        )

        version = await self.predictor.get_version.remote()
        feast_online = await self.feature_lookup.is_available.remote()
        return HealthResponse(
            status="ok",
            model_name=settings.model_name,
            model_version=version,
            mlflow_uri=settings.mlflow_uri,
            feast_online=feast_online,
        )

    @app.get("/model")
    async def model_info(self) -> dict:
        """Return metadata about the currently loaded champion model."""
        info = await self.predictor.get_model_info.remote()
        if info is None:
            raise HTTPException(status_code=404, detail="No model loaded")
        return info

    @app.post("/reload", response_model=ReloadResponse)
    async def reload_model(self) -> ReloadResponse:
        """Hot-reload the champion model from MLflow without downtime."""
        previous, current = await self.predictor.reload.remote()
        return ReloadResponse(
            status="reloaded" if current != previous else "unchanged",
            previous_version=previous,
            current_version=current,
        )

    @app.post("/materialize", response_model=TriggerDagResponse)
    async def trigger_materialize(self) -> TriggerDagResponse:
        """Trigger the Airflow materialization pipeline."""
        return await self._trigger_dag("vroom_forecast_materialize")

    @app.post("/train", response_model=TriggerDagResponse)
    async def trigger_train(self) -> TriggerDagResponse:
        """Trigger the end-to-end ML pipeline (training + promotion)."""
        return await self._trigger_dag("vroom_forecast_pipeline")

    async def _trigger_dag(self, dag_id: str) -> TriggerDagResponse:
        """Trigger an Airflow DAG run via the REST API."""
        import httpx

        if not settings.airflow_url:
            raise HTTPException(status_code=503, detail="Airflow URL not configured")

        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{settings.airflow_url}/api/v1/dags/{dag_id}/dagRuns",
                json={},
                auth=("admin", "admin"),
                timeout=10,
            )

        if res.status_code >= 400:
            detail = (
                res.json().get("detail", res.text)
                if res.headers.get("content-type", "").startswith("application/json")
                else res.text
            )
            raise HTTPException(status_code=res.status_code, detail=detail)

        body = res.json()
        return TriggerDagResponse(
            status="triggered",
            dag_id=dag_id,
            dag_run_id=body.get("dag_run_id"),
        )

    @app.get(
        "/pipelines/{dag_id}/{dag_run_id:path}",
        response_model=DagRunStatus,
    )
    async def get_dag_run_status(self, dag_id: str, dag_run_id: str) -> DagRunStatus:
        """Poll the status of an Airflow DAG run."""
        import httpx

        if not settings.airflow_url:
            raise HTTPException(status_code=503, detail="Airflow URL not configured")

        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{settings.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}",
                auth=("admin", "admin"),
                timeout=10,
            )

        if res.status_code >= 400:
            raise HTTPException(
                status_code=res.status_code,
                detail=f"DAG run not found: {dag_id}/{dag_run_id}",
            )

        body = res.json()
        return DagRunStatus(
            dag_id=dag_id,
            dag_run_id=dag_run_id,
            state=body.get("state", "unknown"),
        )

    @app.get("/stores", response_model=StoreInfoResponse)
    async def store_info(self) -> StoreInfoResponse:
        """Operational info about both feature stores."""

        # Offline store info (run blocking I/O in a thread)
        offline_available = await self.offline_reader.is_available.remote()
        offline_info = StoreDetails(available=offline_available, type="file (Parquet)")
        if offline_available and settings.offline_store_path:
            offline_path = settings.offline_store_path

            def _stat_offline() -> StoreDetails:
                import os

                info = StoreDetails(available=True, type="file (Parquet)")
                try:
                    stat = os.stat(offline_path)
                    info.path = offline_path
                    info.size_bytes = stat.st_size
                    info.last_modified = stat.st_mtime
                except OSError:
                    pass
                return info

            offline_info = await asyncio.to_thread(_stat_offline)

        # Online store info (run blocking Redis calls in a thread)
        feast_online = await self.feature_lookup.is_available.remote()
        online_info = StoreDetails(available=feast_online, type="redis")
        if feast_online and settings.redis_url:
            redis_url = settings.redis_url

            def _stat_online() -> StoreDetails:
                import redis as redis_lib

                info = StoreDetails(available=True, type="redis")
                try:
                    r = redis_lib.from_url(redis_url)
                    try:
                        mem_info = r.info("memory")
                        info.used_memory_human = mem_info.get("used_memory_human", "?")
                        info.keys = r.dbsize()
                        info.redis_url = redis_url
                    finally:
                        r.close()
                except Exception:
                    pass
                return info

            online_info = await asyncio.to_thread(_stat_online)

        return StoreInfoResponse(
            offline_store=offline_info,
            online_store=online_info,
            feature_view=await self._get_feature_view_info(),
        )

    async def _get_feature_view_info(self) -> FeatureViewInfo:
        """Read the feature view definition from Feast, falling back to defaults."""
        try:
            info = await self.feature_lookup.get_feature_view_info.remote()
            if info is not None:
                return FeatureViewInfo(**info)
        except Exception:
            pass
        return FeatureViewInfo()

    @app.get("/events")
    async def events(self) -> StreamingResponse:
        """SSE endpoint — streams model events from Redis pub/sub.

        Subscribes to the `model-promoted` channel which carries both
        promotion events (from the pipeline) and reload events (from the
        Predictor). On each event, fetches the current Predictor state
        and emits a health-changed SSE event.

        Purely event-driven — no polling.
        """
        predictor_handle = self.predictor

        async def event_stream() -> AsyncGenerator[str, None]:
            import redis.asyncio as aioredis

            # Emit current health status on connect
            is_loaded = await predictor_handle.is_loaded.remote()
            is_loading = await predictor_handle.is_loading.remote()
            version = ""
            if is_loaded:
                version = await predictor_handle.get_version.remote()
            if is_loading:
                status = "loading"
            elif is_loaded:
                status = "ok"
            else:
                status = "no_model"
            evt = {
                "type": "health-changed",
                "status": status,
                "model_version": version,
            }
            yield f"data: {json.dumps(evt)}\n\n"

            if settings.redis_url is None:
                while True:
                    await asyncio.sleep(30)
                    yield ": keepalive\n\n"
                return

            r = aioredis.from_url(settings.redis_url)
            pubsub = r.pubsub()
            from serving.model import MODEL_LOADED_CHANNEL

            await pubsub.subscribe(MODEL_LOADED_CHANNEL)

            try:
                while True:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                    if msg and msg["type"] == "message":
                        # Any event on this channel means model state changed
                        is_loaded = await predictor_handle.is_loaded.remote()
                        version = ""
                        if is_loaded:
                            version = await predictor_handle.get_version.remote()
                        evt = {
                            "type": "health-changed",
                            "status": "ok" if is_loaded else "no_model",
                            "model_version": version,
                        }
                        yield f"data: {json.dumps(evt)}\n\n"
                    else:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)
            finally:
                await pubsub.close()
                await r.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/vehicles/events")
    async def vehicle_events(self) -> StreamingResponse:
        """SSE endpoint — streams vehicle-materialized events from Redis pub/sub.

        The UI subscribes via EventSource to get notified when a new arrival's
        features are materialized, replacing polling.
        """

        async def event_stream() -> AsyncGenerator[str, None]:
            import redis.asyncio as aioredis

            if settings.redis_url is None:
                while True:
                    await asyncio.sleep(60)
                return

            r = aioredis.from_url(settings.redis_url)
            pubsub = r.pubsub()
            from serving.model import VEHICLE_MATERIALIZED_CHANNEL

            await pubsub.subscribe(VEHICLE_MATERIALIZED_CHANNEL)

            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        data = json.loads(message["data"])
                        event_data = {
                            "type": "vehicle-materialized",
                            "vehicle_id": data.get("vehicle_id"),
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    else:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)
            finally:
                await pubsub.close()
                await r.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/vehicles", response_model=SaveVehicleResponse)
    async def save_vehicle_endpoint(self, vehicle: VehicleFeatures) -> SaveVehicleResponse:
        """Persist a new vehicle to SQLite and publish a materialization event."""
        vehicle_id, event_published = await asyncio.to_thread(save_vehicle, vehicle)
        return SaveVehicleResponse(
            vehicle_id=vehicle_id, status="saved", event_published=event_published
        )

    @app.delete("/vehicles/{vehicle_id}", response_model=DeleteVehicleResponse)
    async def delete_vehicle_endpoint(self, vehicle_id: int) -> DeleteVehicleResponse:
        """Delete a new arrival vehicle. Fleet vehicles cannot be deleted."""
        deleted = await asyncio.to_thread(delete_vehicle, vehicle_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle {vehicle_id} not found or is a fleet vehicle (cannot delete).",
            )
        return DeleteVehicleResponse(status="deleted", vehicle_id=str(vehicle_id))

    @app.get("/vehicles", response_model=list[VehicleRecord])
    async def list_vehicles_endpoint(self) -> list[VehicleRecord]:
        """List all vehicles from SQLite with reservation counts."""
        return await asyncio.to_thread(list_vehicles)

    @app.get("/vehicles/features", response_model=list[ComputedFeatures])
    async def list_vehicle_features(self) -> list[ComputedFeatures]:
        """Get computed features for all vehicles in a single call.

        Checks the offline store (Parquet) first for all vehicles. Any vehicles
        not found in the offline store are looked up in the online store (Redis)
        as a fallback — these are recently-saved vehicles whose features haven't
        been included in the batch pipeline yet.
        """
        vehicles = await asyncio.to_thread(list_vehicles)
        if not vehicles:
            return []

        all_ids = [v.vehicle_id for v in vehicles]
        results: dict[int, ComputedFeatures] = {}

        # Step 1: Try offline store (Parquet) for all vehicles
        offline_available = await self.offline_reader.is_available.remote()
        if offline_available:
            df = await self.offline_reader.lookup.remote(all_ids)
            if df is not None:
                for _, row in df.iterrows():
                    vid = int(row["vehicle_id"])
                    results[vid] = ComputedFeatures(
                        vehicle_id=vid,
                        technology=int(row["technology"]),
                        num_images=int(row["num_images"]),
                        street_parked=int(row["street_parked"]),
                        description=int(row["description"]),
                        price_diff=float(row["price_diff"]),
                        materialized=True,
                        store="offline",
                    )

        # Step 2: Fallback to online store (Redis) for missing vehicles
        missing_ids = [vid for vid in all_ids if vid not in results]
        if missing_ids:
            feast_online = await self.feature_lookup.is_available.remote()
            if feast_online:
                try:
                    df = await self.feature_lookup.lookup.remote(missing_ids)
                    for _, row in df.iterrows():
                        vid = int(row["vehicle_id"])
                        if pd.isna(row.get("technology")):
                            results[vid] = ComputedFeatures(vehicle_id=vid)
                        else:
                            results[vid] = ComputedFeatures(
                                vehicle_id=vid,
                                technology=int(row["technology"]),
                                num_images=int(row["num_images"]),
                                street_parked=int(row["street_parked"]),
                                description=int(row["description"]),
                                price_diff=float(row["price_diff"]),
                                materialized=True,
                                store="online",
                            )
                except Exception:
                    pass  # vehicles stay missing — not materialized

        return list(results.values())

    @app.get("/vehicles/{vehicle_id}/features", response_model=ComputedFeatures)
    async def get_vehicle_features(self, vehicle_id: int) -> ComputedFeatures:
        """Get computed features for a vehicle.

        Checks the offline store (Parquet) first — this contains all vehicles
        after the batch pipeline has run. Falls back to the online store (Redis)
        for recently-saved vehicles not yet in the Parquet.
        """
        # Try offline store first (Parquet — covers all vehicles post-pipeline)
        offline_available = await self.offline_reader.is_available.remote()
        if offline_available:
            try:
                df = await self.offline_reader.lookup.remote([vehicle_id])
                if df is not None:
                    row = df.iloc[0]
                    return ComputedFeatures(
                        vehicle_id=vehicle_id,
                        technology=int(row["technology"]),
                        num_images=int(row["num_images"]),
                        street_parked=int(row["street_parked"]),
                        description=int(row["description"]),
                        price_diff=float(row["price_diff"]),
                        materialized=True,
                        store="offline",
                    )
            except Exception:
                pass  # fall through to online store

        # Fallback: online store (Redis — for recently-saved vehicles)
        feast_online = await self.feature_lookup.is_available.remote()
        if not feast_online:
            return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)
        try:
            df = await self.feature_lookup.lookup.remote([vehicle_id])
            row = df.iloc[0]
            if pd.isna(row.get("technology")):
                return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)
            return ComputedFeatures(
                vehicle_id=vehicle_id,
                technology=int(row["technology"]),
                num_images=int(row["num_images"]),
                street_parked=int(row["street_parked"]),
                description=int(row["description"]),
                price_diff=float(row["price_diff"]),
                materialized=True,
                store="online",
            )
        except Exception:
            return ComputedFeatures(vehicle_id=vehicle_id, materialized=False)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(self, vehicle: VehicleFeatures) -> PredictionResponse:
        """Predict reservations from raw vehicle features (on-the-fly computation)."""
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(
                status_code=503, detail="No model loaded. Train and promote a model first."
            )
        features_df = await self.feature_computer.compute.remote([vehicle])
        predictions = await self.predictor.predict.remote(features_df)
        version = await self.predictor.get_version.remote()
        return PredictionResponse(
            predicted_reservations=round(predictions[0], 2),
            model_version=version,
        )

    @app.post("/predict/id", response_model=PredictionResponse)
    async def predict_by_id(self, req: VehicleIdRequest) -> PredictionResponse:
        """Predict reservations using pre-computed features from the online store."""
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(
                status_code=503, detail="No model loaded. Train and promote a model first."
            )
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
        """Predict reservations for up to 1000 vehicles in a single request."""
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(
                status_code=503, detail="No model loaded. Train and promote a model first."
            )
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

    async def _run_benchmark(
        self,
        n_iterations: int,
        feature_latencies: list[float],
        predict_latencies: list[float],
        source: str,
    ) -> BenchmarkResponse:
        """Build a BenchmarkResponse from pre-collected latency lists."""
        total_latencies = [f + p for f, p in zip(feature_latencies, predict_latencies, strict=True)]
        version = await self.predictor.get_version.remote()
        arr = np.array(total_latencies)
        return BenchmarkResponse(
            n_iterations=n_iterations,
            avg_latency_ms=round(float(np.mean(arr)), 3),
            p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
            p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
            model_version=version,
            source=source,
            avg_features_ms=round(float(np.mean(feature_latencies)), 3),
            avg_predict_ms=round(float(np.mean(predict_latencies)), 3),
        )

    @app.post("/benchmark", response_model=BenchmarkResponse)
    async def benchmark(self, req: BenchmarkRequest) -> BenchmarkResponse:
        """Benchmark: feature computation (on the fly) + inference."""
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(
                status_code=503, detail="No model loaded. Train and promote a model first."
            )
        # Run feature computation once to get the DataFrame for predict benchmark
        features_df = await self.feature_computer.compute.remote([req.vehicle])
        # Run both benchmark loops inside the actors in parallel (1 .remote() each)
        feature_latencies, predict_latencies = await asyncio.gather(
            self.feature_computer.benchmark_compute.remote([req.vehicle], req.n_iterations),
            self.predictor.benchmark_predict.remote(features_df, req.n_iterations),
        )
        return await self._run_benchmark(
            n_iterations=req.n_iterations,
            feature_latencies=feature_latencies,
            predict_latencies=predict_latencies,
            source="raw",
        )

    @app.post("/benchmark/id", response_model=BenchmarkResponse)
    async def benchmark_by_id(self, req: BenchmarkByIdRequest) -> BenchmarkResponse:
        """Benchmark: feature lookup (online store) + inference."""
        is_loaded = await self.predictor.is_loaded.remote()
        if not is_loaded:
            raise HTTPException(
                status_code=503, detail="No model loaded. Train and promote a model first."
            )
        feast_online = await self.feature_lookup.is_available.remote()
        if not feast_online:
            raise HTTPException(status_code=503, detail="Feature store not configured")
        # Run feature lookup once to get the DataFrame for predict benchmark
        features_df = await self.feature_lookup.lookup.remote([req.vehicle_id])
        # Run both benchmark loops inside the actors in parallel (1 .remote() each)
        feature_latencies, predict_latencies = await asyncio.gather(
            self.feature_lookup.benchmark_lookup.remote([req.vehicle_id], req.n_iterations),
            self.predictor.benchmark_predict.remote(features_df, req.n_iterations),
        )
        return await self._run_benchmark(
            n_iterations=req.n_iterations,
            feature_latencies=feature_latencies,
            predict_latencies=predict_latencies,
            source="online_store",
        )
