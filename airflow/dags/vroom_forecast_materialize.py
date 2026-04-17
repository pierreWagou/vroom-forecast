"""
Vroom Forecast — Feature Materialization DAG

Seeds the database from CSVs (idempotent), computes features, writes to
the Feast offline store (Parquet), and materializes new arrivals to the
online store (Redis).

The host runs seed + Parquet computation as part of `mise run dev` so the
catalog is available immediately. This DAG handles the full pipeline
including the online store (Redis) which requires Docker services.

On completion (success or failure), a notification task publishes an event
to Redis pub/sub so the serving SSE endpoint can push it to the UI —
replacing client-side polling.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

DATA_DIR = "/opt/airflow/data"
DB_PATH = "/feast-data/vehicles.db"
FEAST_REPO = "/opt/airflow/features/feature_repo"
PARQUET_PATH = "/feast-data/vehicle_features.parquet"
PROJECT_DIR = "/opt/airflow"
REDIS_URL = "redis://redis:6379"
PIPELINE_CHANNEL = "vroom-forecast:pipeline-completed"


def _notify_cmd(state: str) -> str:
    """Build a bash command that publishes a pipeline-completed event to Redis.

    Reuses the promotion sub-project's venv (which has the redis client)
    to keep the Airflow image free of ML/data dependencies.
    """
    return (
        f'uv run --project promotion python -c "'
        f"import json, redis; "
        f"r = redis.from_url('{REDIS_URL}'); "
        f"r.publish('{PIPELINE_CHANNEL}', json.dumps({{"
        f"'dag_id': '{{{{ dag.dag_id }}}}', "
        f"'dag_run_id': '{{{{ run_id }}}}', "
        f"'state': '{state}'"
        f"}})); "
        f'r.close()"'
    )


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
        execution_timeout=pendulum.duration(minutes=15),
    )

    # ── Completion notifications ─────────────────────────────────────────
    # These tasks publish to Redis so the UI receives an SSE event instead
    # of polling the Airflow REST API for DAG run status.

    notify_success = BashOperator(
        task_id="notify_success",
        cwd=PROJECT_DIR,
        bash_command=_notify_cmd("success"),
        # Default trigger rule (all_success): runs only when all upstream tasks succeed
        retries=0,
    )

    notify_failure = BashOperator(
        task_id="notify_failure",
        cwd=PROJECT_DIR,
        bash_command=_notify_cmd("failed"),
        trigger_rule=TriggerRule.ONE_FAILED,
        retries=0,
    )

    seed >> materialize >> [notify_success, notify_failure]
