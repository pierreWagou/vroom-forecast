"""
Vroom Forecast — Feature Materialization DAG

Computes features from raw data, writes to the Feast offline store (Parquet),
and materializes to the online store (Redis).

Runs daily at 01:00 UTC — features stay fresh independently of training.
Can also be triggered manually or by an upstream data pipeline.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG

DATA_DIR = "/opt/airflow/data"
FEAST_REPO = "/opt/airflow/features/feature_repo"
PARQUET_PATH = "/feast-data/vehicle_features.parquet"
VEHICLES_DB = "/feast-data/vehicles.db"
PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="vroom_forecast_materialize",
    description="Compute and materialize vehicle features to offline + online stores",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 1 * * *",  # Daily at 01:00 UTC
    catchup=False,
    tags=["ml", "vroom-forecast", "features"],
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
    },
) as dag:
    materialize = BashOperator(
        task_id="materialize",
        cwd=PROJECT_DIR,
        bash_command=(
            f"cd features && uv run python pipeline.py "
            f"--data-dir {DATA_DIR} "
            f"--feast-repo {FEAST_REPO} "
            f"--parquet-path {PARQUET_PATH} "
            f"--vehicles-db {VEHICLES_DB}"
        ),
    )
