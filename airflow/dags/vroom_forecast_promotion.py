"""
Vroom Forecast — Promotion DAG

Compares the candidate model against the current champion and promotes it
if the candidate has a lower cv_mae_mean.

This DAG is event-driven: triggered by the training DAG via
TriggerDagRunOperator after a new model version is tagged as "candidate".
It can also be triggered manually (will resolve the "candidate" alias
from MLflow if no version is provided via DAG conf).
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG

MLFLOW_URI = "http://mlflow:5000"
REDIS_URL = "redis://redis:6379"
PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="vroom_forecast_promotion",
    description="Promote candidate model to champion if it outperforms",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,  # Event-driven only
    catchup=False,
    tags=["ml", "vroom-forecast", "promotion"],
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=2),
    },
) as dag:
    # If triggered with conf.model_version, use it; otherwise fall back to
    # the "candidate" alias in MLflow.
    version_arg = (
        "{% if dag_run.conf and dag_run.conf.model_version %}"
        "--version {{ dag_run.conf.model_version }} "
        "{% endif %}"
    )

    promote = BashOperator(
        task_id="promote",
        cwd=PROJECT_DIR,
        bash_command=(
            f"uv run --project promotion python -m promotion "
            f"{version_arg}"
            f"--mlflow-uri {MLFLOW_URI} "
            f"--redis-url {REDIS_URL}"
        ),
    )
