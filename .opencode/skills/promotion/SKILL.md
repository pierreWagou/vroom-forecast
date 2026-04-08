---
name: promotion
description: Champion/challenger model promotion — MLflow alias management, Redis pub/sub notification
---

## Role

You are a platform/MLOps engineer maintaining the model promotion gate.

## Overview

Independent uv project (`promotion/`) that compares a candidate model against the
current champion and promotes it if the candidate has a lower `cv_mae_mean`.
This project is intentionally separate from training — a different team could own it.

## How It Works

1. Resolves candidate by explicit `--version` flag or the `candidate` MLflow alias
2. Compares `cv_mae_mean` metric against the current `champion`
3. If candidate is better: moves `champion` alias to the new version
4. Publishes `model-promoted` event to Redis pub/sub (`vroom-forecast:model-promoted`)
5. Always exits 0 (prints `promoted` or `retained`); exits 1 only for unexpected errors

## Rules

- Promotion logic: lower `cv_mae_mean` wins, strict improvement only
- Never promote a model that hasn't been compared against the current champion
- On successful promotion, publish event to Redis `vroom-forecast:model-promoted` channel
- Exit 0 for promoted/retained (valid outcomes); exit 1 for unexpected errors (so Airflow marks the task as failed)

## File Layout

```
promotion/
  __init__.py
  __main__.py        # CLI entry point (argparse)
  promote.py         # Core promotion logic
  tests/
    test_promote.py  # Unit tests
  pyproject.toml
  uv.lock
  ty.toml
```

## Dependencies

mlflow, redis. Dev: pytest.

## Run Locally

```bash
uv run --project promotion python -m promotion \
    --mlflow-uri http://localhost:5001 \
    --redis-url redis://localhost:6379
```

## Tests

```bash
cd promotion && uv run --group dev pytest tests/ -q
```

## Standards

- Format/lint: ruff (root `ruff.toml`)
- Type check: `uvx ty check --python promotion/.venv promotion/`
