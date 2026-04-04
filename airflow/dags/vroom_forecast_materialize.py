"""
Vroom Forecast — Feature Materialization DAG

Seeds the database from CSVs (idempotent), computes features, writes to
the Feast offline store (Parquet), and materializes to the online store (Redis).

Runs daily at 01:00 UTC — features stay fresh independently of training.
Can also be triggered manually or by an upstream data pipeline.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG

DATA_DIR = "/opt/airflow/data"
DB_PATH = "/feast-data/vehicles.db"
FEAST_REPO = "/opt/airflow/features/feature_repo"
PARQUET_PATH = "/feast-data/vehicle_features.parquet"
PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="vroom_forecast_materialize",
    description="Seed database, compute and materialize vehicle features",
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
    seed = BashOperator(
        task_id="seed",
        cwd=PROJECT_DIR,
        bash_command=(f"cd features && uv run python seed.py --data-dir {DATA_DIR} --db {DB_PATH}"),
    )

    materialize = BashOperator(
        task_id="materialize",
        cwd=PROJECT_DIR,
        bash_command=(
            f"cd features && uv run python pipeline.py "
            f"--db {DB_PATH} "
            f"--feast-repo {FEAST_REPO} "
            f"--parquet-path {PARQUET_PATH}"
        ),
    )

    seed >> materialize
