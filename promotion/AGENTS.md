# Promotion — Agent Instructions

Read README.md for full context.

You are a platform/MLOps engineer maintaining the model promotion gate.

- This project is intentionally separate from training — different team could own it
- Dependencies: mlflow + redis (for pub/sub notification on promotion)
- Promotion logic: lower `cv_mae_mean` wins, strict improvement only
- Never promote a model that hasn't been compared against the current champion
- On successful promotion, publish event to Redis `vroom-forecast:model-promoted` channel
- Exit 0 for promoted/retained (valid outcomes); exit 1 for unexpected errors (so Airflow marks the task as failed)
- Type-checked with ty; formatted with ruff
