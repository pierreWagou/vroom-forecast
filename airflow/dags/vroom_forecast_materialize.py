"""
Vroom Forecast — Feature Materialization DAG

Seeds the database from CSVs (idempotent), computes features, writes to
the Feast offline store (Parquet), and materializes new arrivals to the
online store (Redis).

The host runs seed + Parquet computation as part of `mise run dev` so the
catalog is available immediately. This DAG handles the full pipeline
including the online store (Redis) which requires Docker services.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG, Dataset

DATA_DIR = "/opt/airflow/data"
DB_PATH = "/feast-data/vehicles.db"
FEAST_REPO = "/opt/airflow/features/feature_repo"
PARQUET_PATH = "/feast-data/vehicle_features.parquet"
PROJECT_DIR = "/opt/airflow"

# Dataset marker for cross-DAG dependency
FEATURES_DATASET = Dataset("file:///feast-data/vehicle_features.parquet")

with DAG(
    dag_id="vroom_forecast_materialize",
    description="Seed database and materialize vehicle features",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,  # Manual trigger only
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
        execution_timeout=pendulum.duration(minutes=10),
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
        outlets=[FEATURES_DATASET],
        execution_timeout=pendulum.duration(minutes=15),
    )

    seed >> materialize
