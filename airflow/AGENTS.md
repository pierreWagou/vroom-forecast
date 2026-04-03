# Airflow — Agent Instructions

Read README.md for full context.

You are an MLOps engineer maintaining pipeline orchestration.

- Airflow is orchestration only — NO ML dependencies in the Airflow image
- Tasks run via `BashOperator` + `uv run --project <project> python -m <module>`
- DAGs are Python files but not type-checked (Airflow deps not in local venv)
- Training DAG triggers promotion DAG via `TriggerDagRunOperator`
- Model version is passed between DAGs via XCom (stdout) and `dag_run.conf`
- Keep the Dockerfile minimal: just uv + sub-project source files
