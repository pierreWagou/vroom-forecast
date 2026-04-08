---
name: local-dev
description: Local development — Docker services, mprocs, pipeline triggers, ports, and dev tools setup
---

## Local Development

Start all services:
```bash
mprocs                  # MLflow, Redis, Airflow, Serving, UI, Jupyter, Docs
docker compose up       # Just Docker services (no UI, Jupyter, or Docs)
```

## Running Pipelines Locally

> **Note:** Feature materialization and offline-store training require Docker
> services (Redis, MLflow). Use `docker compose up -d mlflow redis` first,
> or run the full pipeline via Airflow (see below).

Feature seeding + materialization (requires Docker services running):
```bash
# Set Feast env vars for local dev (feature_store.yaml uses these)
export FEAST_REGISTRY=../feast-data/registry.db
export FEAST_REDIS=localhost:6379

# Seed the database from CSVs (one-time, idempotent)
cd features && uv run python seed.py \
    --data-dir ../data \
    --db ../feast-data/vehicles.db

# Compute and materialize features
cd features && uv run python pipeline.py \
    --db ../feast-data/vehicles.db \
    --feast-repo feature_repo \
    --parquet-path ../feast-data/vehicle_features.parquet
```

Training (requires materialization first):
```bash
uv run --project training python -m training \
    --feature-store feast-data/vehicle_features.parquet \
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

## Running via Airflow (Recommended)

The easiest path is to use Airflow inside Docker, which handles seeding,
materialization, training, and promotion in the correct order:

```bash
# Trigger the full pipeline (materialize -> training -> promotion auto-chains)
docker compose exec airflow airflow dags trigger vroom_forecast_materialize

# Or trigger individual steps
docker compose exec airflow airflow dags trigger vroom_forecast_training
docker compose exec airflow airflow dags trigger vroom_forecast_promotion
```

Airflow credentials: `admin` / `admin`

## Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `vroom_forecast_materialize` | `0 1 * * *` (daily) | Seed DB + compute features -> Parquet (all) + Redis (new arrivals) |
| `vroom_forecast_training` | Dataset-driven (after materialize) | Train from offline store -> register candidate |
| `vroom_forecast_promotion` | None (event-driven) | Compare candidate vs champion -> promote |

## Ports

| Service | Port |
|---------|------|
| MLflow UI | http://localhost:5001 |
| Airflow UI | http://localhost:8080 |
| Ray Serve API | http://localhost:8000 |
| Ray Serve docs | http://localhost:8000/docs |
| Ray Dashboard | http://localhost:8265 |
| Redis | localhost:6379 |
| Redis Insight | http://localhost:5540 |
| UI (Next.js) | http://localhost:3000 |
| Docs (MkDocs) | http://localhost:8100 |
| Jupyter | http://localhost:8888 |

## Dev Tools Setup

```bash
uvx pre-commit install   # Set up git hooks (one-time)
uvx pre-commit run --all-files  # Run all checks
```
