# Airflow

Orchestration layer for ML pipelines. Contains the Dockerfile and DAG
definitions. No ML dependencies — pipelines run via `BashOperator` + `uv run`.

## DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `vroom_forecast_materialize` | `0 1 * * *` (daily 01:00 UTC) | Compute features from CSVs + SQLite, write to Parquet + Redis |
| `vroom_forecast_training` | `0 2 * * 0` (Sundays 02:00 UTC) | Train model from offline store, register as candidate, trigger promotion |
| `vroom_forecast_promotion` | None (event-driven + manual) | Compare candidate vs champion, promote if better, notify via Redis |

## How it works

Airflow doesn't install any ML dependencies. Each task runs:

```bash
uv run --project <project> python -m <module> [args]
```

uv creates an isolated venv inside the container on first run.

The training DAG triggers the promotion DAG via `TriggerDagRunOperator`,
passing the model version through `dag_run.conf`.

## Key files

- `Dockerfile` — Extends `apache/airflow:2.10.5-python3.12`, adds `uv`, copies sub-projects
- `dags/vroom_forecast_materialize.py` — Feature materialization DAG
- `dags/vroom_forecast_training.py` — Training DAG
- `dags/vroom_forecast_promotion.py` — Promotion DAG

## Credentials

`airflow standalone` generates an admin password on first start:

```bash
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```
