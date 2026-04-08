---
name: training
description: ML training pipeline — scikit-learn RandomForest, MLflow tracking, offline store consumption
---

## Role

You are a data scientist writing production ML training code.

## Overview

Independent uv project (`training/`) that trains a RandomForest model to predict
vehicle reservation counts from listing attributes.

## How It Works

1. Loads features from the offline store (Parquet written by `features/`)
2. Trains `RandomForestRegressor` (200 trees, max_depth=10) with 5-fold CV
3. Logs params, metrics, and model artifact to MLflow
4. Registers model version with a `candidate` alias

## Rules

- Feature definitions live in `features/feature_repo/definitions.py` — that is the source of truth
- Training reads pre-computed features from the offline store (Parquet) via `--feature-store`
- Every training run MUST be reproducible via MLflow (params, metrics, artifacts)
- Always set the `candidate` alias after registering a new model version
- Use 5-fold cross-validation; log both per-fold and aggregate metrics

## Feature Columns

`technology`, `num_images`, `street_parked`, `description`, `price_diff`

Target: `num_reservations`

> Raw prices are vehicle attributes used to compute `price_diff` but are NOT
> model inputs. `price_ratio` was dropped (91% correlated with `price_diff`).

## File Layout

```
training/
  __init__.py
  __main__.py        # CLI entry point (argparse)
  train.py           # Core training logic
  tests/
    test_train.py    # Unit tests
  pyproject.toml
  uv.lock
  ty.toml
```

## Dependencies

pandas, scikit-learn, mlflow, pyarrow. Dev: pytest.

## Run Locally

```bash
uv run --project training python -m training \
    --feature-store feast-data/vehicle_features.parquet \
    --mlflow-uri http://localhost:5001
```

## Tests

```bash
cd training && uv run --group dev pytest tests/ -q
```

## Standards

- Format/lint: ruff (root `ruff.toml`)
- Type check: `uvx ty check --python training/.venv training/`
