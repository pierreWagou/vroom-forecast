"""
Vroom Forecast — Training DAG

Trains a Random Forest model using features from the offline store and
registers it in MLflow with the "candidate" alias.

Independent pipeline — can be triggered standalone or chained via the
orchestrator DAG (vroom_forecast_pipeline).
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG

MLFLOW_URI = "http://mlflow:5000"
FEATURE_STORE = "/feast-data/vehicle_features.parquet"
PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="vroom_forecast_training",
    description="Train the vroom-forecast model and tag it as candidate",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,  # Manual or called by orchestrator
    catchup=False,
    tags=["ml", "vroom-forecast", "training"],
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
    },
) as dag:
    train = BashOperator(
        task_id="train",
        cwd=PROJECT_DIR,
        bash_command=(
            f"uv run --project training python -m training "
            f"--feature-store {FEATURE_STORE} "
            f"--mlflow-uri {MLFLOW_URI}"
        ),
        execution_timeout=pendulum.duration(minutes=30),
    )
