"""
Vroom Forecast — Pipeline Orchestrator DAG

Chains training and promotion into a single end-to-end pipeline.
Waits for each stage to complete before starting the next.

This is the DAG triggered by the UI "Train" button. The individual
training and promotion DAGs remain independently triggerable from
the Airflow UI for debugging or re-runs.
"""

from __future__ import annotations

import pendulum
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

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

    trigger_training >> trigger_promotion
