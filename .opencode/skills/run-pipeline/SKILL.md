---
name: run-pipeline
description: How to run services locally — all pipelines, Docker services, and dev tools
---

## Local Development

Start all services:
```bash
mprocs                  # MLflow, Redis, Airflow, Serving, UI, Jupyter
docker compose up       # Just Docker services (no UI or Jupyter)
```

## Running Pipelines Locally

Feature materialization:
```bash
cd features && uv run python pipeline.py \
    --data-dir ../data \
    --feast-repo feature_repo
```

Training (from offline store):
```bash
uv run --project training python -m training \
    --feature-store /feast-data/vehicle_features.parquet \
    --mlflow-uri http://localhost:5001
```

Training (from CSV fallback):
```bash
uv run --project training python -m training \
    --data-dir data \
    --mlflow-uri http://localhost:5001
```

Promotion:
```bash
uv run --project promotion python -m promotion \
    --mlflow-uri http://localhost:5001 \
    --redis-url redis://localhost:6379
```

Serving (Ray Serve):
```bash
uv run --project serving python -m serving
```

## Running via Airflow

```bash
docker compose exec airflow airflow dags trigger vroom_forecast_materialize
docker compose exec airflow airflow dags trigger vroom_forecast_training
docker compose exec airflow airflow dags trigger vroom_forecast_promotion
```

## Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `vroom_forecast_materialize` | `0 1 * * *` (daily) | Compute features → Parquet + Redis |
| `vroom_forecast_training` | `0 2 * * 0` (Sundays) | Train from offline store → register candidate |
| `vroom_forecast_promotion` | None (event-driven) | Compare candidate vs champion → promote |

## Ports

| Service | Port |
|---------|------|
| MLflow UI | http://localhost:5001 |
| Airflow UI | http://localhost:8080 |
| Serving API | http://localhost:8000 |
| Serving docs | http://localhost:8000/docs |
| Ray Dashboard | http://localhost:8265 |
| Redis | localhost:6379 |
| Redis Insight | http://localhost:5540 |
| UI (Next.js) | http://localhost:3000 |
| Jupyter | http://localhost:8888 |

## Dev Tools Setup

```bash
uvx pre-commit install   # Set up git hooks (one-time)
uvx pre-commit run --all-files  # Run all checks
```
