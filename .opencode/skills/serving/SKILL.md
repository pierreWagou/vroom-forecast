---
name: serving
description: Ray Serve prediction API — FastAPI ingress, Feast feature lookup, MLflow model loading, deployment composition
---

## Role

You are a backend engineer building a production prediction API with Ray Serve.

## Overview

Independent uv project (`serving/`) providing the prediction API via Ray Serve
with a FastAPI ingress. Multiple Ray Serve deployments handle feature computation,
feature lookup, model inference, and feature materialization.

## Rules

- FeatureMaterializer is a Ray actor (not a deployment) for real-time feature materialization
- Feature routing: offline store (Parquet) checked first, online store (Redis) as fallback
- Model reload: `Predictor.reload()` triggered via `/reload` endpoint or Redis pub/sub
- CORS is enabled for the UI; review `allow_origins` before production
- Health endpoint is used by Docker healthcheck — keep it fast
- Schemas in `schemas.py` are the API contract; changes require coordination with `ui/`

## Architecture

| Deployment | Role |
|---|---|
| `VroomForecastApp` | FastAPI ingress — routes, SSE, vehicle CRUD |
| `Predictor` | Loads champion model from MLflow, runs inference |
| `FeatureComputer` | On-the-fly feature computation (price_diff) |
| `FeatureLookup` | Feast/Redis online store lookup |
| `OfflineFeatureReader` | Reads all features from Parquet offline store |
| `FeatureMaterializer` | Ray actor — writes features to Redis + Parquet on vehicle save |
| `ModelReloadListener` | Redis pub/sub listener — triggers Predictor reload on promotion |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + model version + Feast status |
| `POST` | `/reload` | Hot-reload champion model from MLflow |
| `POST` | `/predict` | Single prediction from raw attributes |
| `POST` | `/predict/id` | Prediction by vehicle ID (online store) |
| `POST` | `/predict/batch` | Batch prediction (up to 1,000 vehicles) |
| `POST` | `/benchmark` | Latency benchmark: raw features path |
| `POST` | `/benchmark/id` | Latency benchmark: online store path |
| `POST` | `/vehicles` | Save a vehicle to the catalog |
| `DELETE` | `/vehicles/{id}` | Delete a new arrival vehicle |
| `GET` | `/vehicles` | List all vehicles |
| `GET` | `/vehicles/features` | Batch: features for all vehicles |
| `GET` | `/vehicles/{id}/features` | Features for one vehicle |
| `GET` | `/stores` | Offline/online store operational info |
| `GET` | `/model` | Champion model metadata (version, metrics, feature importances) |
| `POST` | `/materialize` | Trigger the Airflow materialization pipeline |
| `POST` | `/train` | Trigger the end-to-end ML pipeline (training + promotion) |
| `GET` | `/events` | SSE stream for model promotion events |
| `GET` | `/vehicles/events` | SSE stream for vehicle materialization events |
| `GET` | `/pipelines/events` | SSE stream for Airflow DAG completion events |

## Configuration

Env vars with `SERVING_` prefix (pydantic-settings):

| Var | Default | Description |
|---|---|---|
| `SERVING_MLFLOW_URI` | `http://localhost:5001` | MLflow tracking URI |
| `SERVING_MODEL_NAME` | `vroom-forecast` | Registered model name |
| `SERVING_HOST` | `0.0.0.0` | Bind host |
| `SERVING_PORT` | `8000` | Bind port |
| `SERVING_FEAST_REPO` | None | Path to Feast repo |
| `SERVING_REDIS_URL` | None | Redis connection URL |
| `SERVING_DB_PATH` | `/feast-data/vehicles.db` | SQLite database path |
| `SERVING_OFFLINE_STORE_PATH` | None | Parquet offline store |
| `SERVING_AIRFLOW_URL` | None | Airflow REST API base URL |

## File Layout

```
serving/
  __init__.py
  __main__.py        # Ray Serve entry point
  app.py             # FastAPI ingress (VroomForecastApp)
  config.py          # Pydantic settings
  features.py        # FeatureComputer, FeatureLookup, OfflineFeatureReader, FeatureMaterializer
  model.py           # Predictor, ModelReloadListener
  schemas.py         # Pydantic request/response schemas
  vehicles.py        # Vehicle CRUD (SQLite)
  Dockerfile
  tests/
    test_serving.py  # Unit tests
  pyproject.toml
  uv.lock
  ty.toml
```

## Dependencies

ray[serve], fastapi, pydantic-settings, mlflow, scikit-learn, pandas, numpy,
feast[redis], redis. Dev: pytest.

## Run Locally

```bash
uv run --project serving python -m serving
```

## Tests

```bash
cd serving && uv run --group dev pytest tests/ -q
```

## Standards

- Format/lint: ruff (root `ruff.toml`)
- Type check: `uvx ty check --python serving/.venv --config-file serving/ty.toml serving/ --exclude serving/tests/`
- `ty.toml` excludes tests, ignores `unresolved-attribute` (Ray decorators), warns on `unresolved-import` (Feast repo path)
