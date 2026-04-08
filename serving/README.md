# Serving API

Ray Serve prediction service for vroom-forecast. FastAPI ingress with
Ray Serve deployments for model inference, feature computation, and
online store lookup.

## Architecture

```mermaid
graph TB
    subgraph Ray Serve
        ING[VroomForecastApp<br/>FastAPI ingress]
        FC[FeatureComputer]
        FL[FeatureLookup]
        OFR[OfflineFeatureReader]
        PRED[Predictor]
        MAT[FeatureMaterializer<br/>Ray actor]
    end

    Client -->|HTTP| ING
    ING -->|/predict| FC -->|DataFrame| PRED
    ING -->|/predict/id| FL -->|DataFrame| PRED
    ING -->|/vehicles/features| OFR -->|offline first| ING
    ING -->|/vehicles/features fallback| FL
    ING -->|/vehicles POST| DB[(SQLite)]
    DB -->|Redis pub/sub| MAT -->|write_to_online_store| Redis[(Redis)]
    PQ[Parquet] --> OFR
    Redis --> FL
    MLflow[(MLflow)] -->|load champion| PRED
```

## Running

```bash
# Local:
uv run --project serving python -m serving

# Docker:
docker compose up serving    # port 8000 + Ray dashboard on 8265
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model info + Feast online status |
| POST | `/reload` | Hot-reload champion model from MLflow |
| POST | `/predict` | Single prediction from raw attributes (on-the-fly features) |
| POST | `/predict/id` | Prediction by vehicle ID (features from online store) |
| POST | `/predict/batch` | Batch prediction (up to 1000, on-the-fly features) |
| POST | `/benchmark` | Latency benchmark: on-the-fly features + inference |
| POST | `/benchmark/id` | Latency benchmark: online store lookup + inference |
| POST | `/vehicles` | Save a vehicle to SQLite (emits Redis event for materialization) |
| DELETE | `/vehicles/{id}` | Delete a new arrival vehicle |
| GET | `/vehicles` | List all vehicles |
| GET | `/vehicles/features` | Batch: get features for all vehicles (offline + online) |
| GET | `/vehicles/{id}/features` | Get computed features for one vehicle |
| GET | `/stores` | Operational info about offline and online stores |
| GET | `/events` | SSE stream for model promotion events |
| GET | `/vehicles/events` | SSE stream for vehicle materialization events |

Interactive docs at `http://localhost:8000/docs`.

A [Bruno](https://www.usebruno.com/) API collection is included at the repo root in `bruno/` with
pre-filled requests for every endpoint.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVING_MLFLOW_URI` | `http://localhost:5001` | MLflow tracking server |
| `SERVING_MODEL_NAME` | `vroom-forecast` | Registered model name |
| `SERVING_HOST` | `0.0.0.0` | Bind address |
| `SERVING_PORT` | `8000` | Bind port |
| `SERVING_FEAST_REPO` | None | Path to Feast feature repo |
| `SERVING_REDIS_URL` | None | Redis URL for pub/sub + model reload |
| `SERVING_DB_PATH` | `/feast-data/vehicles.db` | SQLite path for vehicle persistence |
| `SERVING_OFFLINE_STORE_PATH` | None | Parquet path for offline feature store |
