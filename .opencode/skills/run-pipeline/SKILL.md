---
name: run-pipeline
description: How to run training, promotion, and exploration pipelines — locally with uv, via Airflow, or with docker compose
---

## Local Development

Start all services:
```bash
mprocs                  # MLflow (5001) + Airflow (8080) + Serving (8000) + Jupyter (8888)
docker compose up       # Just MLflow + Airflow + Serving
```

## Running Pipelines Locally

Training (registers a model version, sets `candidate` alias):
```bash
uv run --project training python -m training \
    --data-dir data \
    --mlflow-uri http://localhost:5001
```

Promotion (compares candidate vs champion — uses its own uv project):
```bash
# By alias (default: resolves "candidate" from MLflow):
uv run --project promotion python -m promotion \
    --mlflow-uri http://localhost:5001

# By explicit version:
uv run --project promotion python -m promotion \
    --version 5 \
    --mlflow-uri http://localhost:5001
```

Serving (FastAPI — runs locally against MLflow):
```bash
uv run --project serving python -m serving
# Or with env var:
SERVING_MLFLOW_URI=http://localhost:5001 uv run --project serving python -m serving
```

Exploration (Jupyter notebook):
```bash
uv run --project exploration jupyter notebook
```

## Running Pipelines via Airflow

Trigger training (cascades to promotion via TriggerDagRunOperator):
```bash
docker compose exec airflow airflow dags trigger vroom_forecast_training
```

Trigger promotion standalone:
```bash
# With explicit version via conf:
docker compose exec airflow airflow dags trigger vroom_forecast_promotion \
    --conf '{"model_version": "5"}'

# Without conf (resolves "candidate" alias from MLflow):
docker compose exec airflow airflow dags trigger vroom_forecast_promotion
```

## Airflow DAGs

| DAG | Schedule | Trigger |
|-----|----------|---------|
| `vroom_forecast_training` | `0 2 * * 0` (Sundays 02:00 UTC) | Scheduled + manual |
| `vroom_forecast_promotion` | None | Event-driven (from training DAG) + manual |

## Inside the Airflow Container

The Airflow image has `uv` but no ML deps. Each BashOperator task runs:
```bash
uv run --project <project> python -m <module> [args]
```
uv creates and caches a venv inside the container on first run.

## Ports

| Service | Port |
|---------|------|
| MLflow UI | http://localhost:5001 |
| Airflow UI | http://localhost:8080 (admin/admin) |
| Serving API | http://localhost:8000 |
| Serving docs | http://localhost:8000/docs |
| Jupyter | http://localhost:8888 |

## Dev Tools Setup (root project)

```bash
uv sync                     # Install ruff, ty, pre-commit
uv run pre-commit install   # Set up git hooks
```
