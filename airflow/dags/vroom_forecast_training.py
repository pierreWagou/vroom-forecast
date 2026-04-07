"""
Vroom Forecast — Training DAG

Trains a Random Forest model using features from the offline store, registers
it in MLflow with the "candidate" alias, then triggers the promotion DAG.

Uses Airflow Datasets for cross-DAG dependency: this DAG is triggered
whenever the materialize DAG produces fresh features (typically daily).
Can also be triggered manually.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG, Dataset

MLFLOW_URI = "http://mlflow:5000"
FEATURE_STORE = "/feast-data/vehicle_features.parquet"
PROJECT_DIR = "/opt/airflow"

# Cross-DAG dependency: only train when features have been refreshed
FEATURES_DATASET = Dataset("file:///feast-data/vehicle_features.parquet")

with DAG(
    dag_id="vroom_forecast_training",
    description="Train the vroom-forecast model and tag it as candidate",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=[FEATURES_DATASET],
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
        # Last line of stdout (the model version) is pushed to XCom
        do_xcom_push=True,
        execution_timeout=pendulum.duration(minutes=30),
    )

    trigger_promotion = TriggerDagRunOperator(
        task_id="trigger_promotion",
        trigger_dag_id="vroom_forecast_promotion",
        conf={"model_version": "{{ ti.xcom_pull(task_ids='train') | trim }}"},
        wait_for_completion=False,
    )

    train >> trigger_promotion
