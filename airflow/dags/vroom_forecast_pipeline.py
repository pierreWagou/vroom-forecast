"""
Vroom Forecast — Pipeline Orchestrator DAG

Chains training and promotion into a single end-to-end pipeline.
Waits for each stage to complete before starting the next.

This is the DAG triggered by the UI "Train" button. The individual
training and promotion DAGs remain independently triggerable from
the Airflow UI for debugging or re-runs.

On completion (success or failure), a notification task publishes an event
to Redis pub/sub so the serving SSE endpoint can push it to the UI —
replacing client-side polling.
"""

from __future__ import annotations

import pendulum
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

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
    dag_id="vroom_forecast_pipeline",
    description="End-to-end ML pipeline: train → promote",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,  # Manual trigger only (UI button or CLI)
    catchup=False,
    tags=["ml", "vroom-forecast", "pipeline"],
    default_args={
        "owner": "mlops",
        "retries": 0,
    },
) as dag:
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="vroom_forecast_training",
        wait_for_completion=True,
        deferrable=True,
        poke_interval=5,
        execution_timeout=pendulum.duration(minutes=45),
    )

    trigger_promotion = TriggerDagRunOperator(
        task_id="trigger_promotion",
        trigger_dag_id="vroom_forecast_promotion",
        wait_for_completion=True,
        deferrable=True,
        poke_interval=5,
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

    trigger_training >> trigger_promotion >> [notify_success, notify_failure]
