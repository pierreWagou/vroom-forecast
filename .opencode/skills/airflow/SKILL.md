---
name: airflow
description: Pipeline orchestration — DAG definitions, BashOperator + uv isolation, Docker build, scheduling
---

## Role

You are an MLOps engineer maintaining pipeline orchestration.

## Overview

Orchestration layer (`airflow/`) for ML pipelines. Contains the Dockerfile and
DAG definitions. No ML dependencies — pipelines run via `BashOperator` + `uv run`
in isolated sub-project environments.

## Rules

- Airflow is orchestration only — NO ML dependencies in the Airflow image
- Tasks run via `BashOperator` + `uv run --project <project> python -m <module>`
- DAGs are Python files but not type-checked (Airflow deps not in local venv)
- Training DAG triggers promotion DAG via `TriggerDagRunOperator`
- Model version is passed between DAGs via XCom (stdout) and `dag_run.conf`
- Keep the Dockerfile minimal: just uv + sub-project source files

## DAGs

| DAG | Schedule | Description |
|---|---|---|
| `vroom_forecast_materialize` | Manual | `seed` -> `materialize`: seed DB + compute features, write Parquet + Redis |
| `vroom_forecast_training` | Manual | `train`: train from offline store, register candidate |
| `vroom_forecast_promotion` | Manual | `promote`: compare candidate vs champion, promote if better |
| `vroom_forecast_pipeline` | Manual | `trigger_training` -> `trigger_promotion`: orchestrator (UI "Train" button) |

## Dependency Chaining

Training explicitly triggers the promotion DAG via `TriggerDagRunOperator`.
The pipeline DAG orchestrates training → promotion in sequence.

## File Layout

```
airflow/
  Dockerfile                              # apache/airflow:2.10.5-python3.12 + uv
  dags/
    vroom_forecast_materialize.py         # Seed + materialize DAG
    vroom_forecast_training.py            # Training DAG
    vroom_forecast_promotion.py           # Promotion DAG
    vroom_forecast_pipeline.py            # Pipeline orchestrator DAG
```

## Docker Build

Base image: `apache/airflow:2.10.5-python3.12`. Installs uv, copies `training/`,
`promotion/`, `features/` sub-projects, pre-syncs all deps at build time so DAG
tasks start instantly.

## Trigger Pipelines

```bash
# Full pipeline (materialize features)
docker compose exec airflow airflow dags trigger vroom_forecast_materialize

# Training + promotion (pipeline orchestrator)
docker compose exec airflow airflow dags trigger vroom_forecast_pipeline

# Individual steps
docker compose exec airflow airflow dags trigger vroom_forecast_training
docker compose exec airflow airflow dags trigger vroom_forecast_promotion
```

Airflow UI: http://localhost:8080 — credentials: `admin` / `admin`

## Key Principle

Airflow is orchestration-only. It has zero ML dependencies. Each task runs
`uv run --project <subproject>` so training, promotion, and features each use
their own isolated venv.
