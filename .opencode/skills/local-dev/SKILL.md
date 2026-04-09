---
name: local-dev
description: Local development — mise tasks, Docker services, mprocs, pipeline triggers, ports, and dev tools setup
---

## mise — Task Runner & Tool Manager

All dev tools and tasks are defined in `mise.toml`. Run `mise tasks ls` to list them.

```bash
mise install          # Install pinned tools (Python 3.12, Node LTS, uv, mprocs)
mise run setup        # Bootstrap all deps + pre-commit hooks
mise run dev          # Start everything (mprocs)
mise run check        # Full CI check: lint + type check + tests (pre-commit)
mise run pipeline     # Full ML pipeline: seed → materialize → train → promote
```

## Local Development

Start all services:
```bash
mise run dev            # Recommended: uses mprocs
docker compose up       # Alternative: just Docker services (no UI, Jupyter, or Docs)
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
# Trigger the full pipeline (materialize features)
docker compose exec airflow airflow dags trigger vroom_forecast_materialize

# Training + promotion (pipeline orchestrator)
docker compose exec airflow airflow dags trigger vroom_forecast_pipeline

# Or trigger individual steps
docker compose exec airflow airflow dags trigger vroom_forecast_training
docker compose exec airflow airflow dags trigger vroom_forecast_promotion
```

Airflow credentials: `admin` / `admin`

## Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `vroom_forecast_materialize` | Manual | Seed DB + compute features -> Parquet (all) + Redis (new arrivals) |
| `vroom_forecast_training` | Manual | Train from offline store -> register candidate |
| `vroom_forecast_promotion` | Manual | Compare candidate vs champion -> promote |
| `vroom_forecast_pipeline` | Manual | Orchestrator: training -> promotion (UI "Train" button) |

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
