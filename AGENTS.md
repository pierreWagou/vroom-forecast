# Vroom Forecast — MLOps Take-Home

## Context

Take-home exercise for a **Staff MLOps Engineer** position at **Turo** (Paris).
Turo is the world's largest car-sharing marketplace.

For tech stack priorities and design principles, load the `turo-context` skill.
For running pipelines and local dev, load the `run-pipeline` skill.

## Project Structure

```
training/          # ML training pipeline (python -m training)
  pyproject.toml   #   Own uv project — pandas, sklearn, mlflow, numpy
promotion/         # Champion/challenger promotion (python -m promotion)
  pyproject.toml   #   Own uv project — mlflow only
serving/           # FastAPI prediction service (python -m serving)
  pyproject.toml   #   Own uv project — fastapi, uvicorn, mlflow, sklearn
  Dockerfile       #   Standalone container
exploration/       # EDA notebook (Jupytext-synced)
  pyproject.toml   #   Own uv project — training deps + matplotlib, seaborn, jupyter
airflow/           # Airflow Dockerfile + DAGs
  dags/            #   - vroom_forecast_training.py (scheduled + manual)
                   #   - vroom_forecast_promotion.py (event-driven)
  Dockerfile       #   uv + project files, no ML deps
data/              # CSV datasets (vehicles + reservations)
docker-compose.yml # MLflow + Airflow + Serving (SequentialExecutor + SQLite)
mprocs.yaml        # Local dev: mlflow, airflow, serving, jupyter
pyproject.toml     # Root — dev tools only (ruff, ty, pre-commit)
```

## Dependency Management

Each sub-project is a **fully independent uv project** with its own
`pyproject.toml`, `uv.lock`, and `.venv`. No shared workspace.

```bash
uv run --project training   python -m training    # uses training/.venv
uv run --project promotion  python -m promotion   # uses promotion/.venv
uv run --project serving    python -m serving     # uses serving/.venv
uv run --project exploration jupyter notebook      # uses exploration/.venv
```

Pipelines run as proper modules: `uv run --project <dir> python -m <module>`.
Airflow uses BashOperator to invoke these — no ML deps in the Airflow image.
