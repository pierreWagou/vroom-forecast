---
name: features
description: Feast feature store — offline (Parquet) and online (Redis) stores, feature definitions, materialization pipeline
---

## Role

You are a data/platform engineer maintaining the feature store.

## Overview

Independent uv project (`features/`) that defines and materializes all vehicle
features using Feast. Single source of truth for feature engineering consumed by
both training and serving.

## Rules

- Feature definitions in `feature_repo/definitions.py` are the single source of truth
- Training and serving MUST use features from this store — no local feature engineering
- The feature pipeline computes derived features (price_diff, price_ratio) once
- The batch pipeline writes to Parquet (all vehicles) and Redis (new arrivals only via `write_to_online_store`)
- The real-time FeatureMaterializer (in serving) writes individual vehicles to Redis on save
- `num_reservations` is nullable: NULL for new arrivals (no observation), int for fleet (observed count, including 0)
- Adding a new feature requires updating: `definitions.py`, `pipeline.py`, and downstream consumers

## Feature Schema

| Feature | Type | Description |
|---|---|---|
| `technology` | Int64 | Instant-bookable tech package (0/1) |
| `num_images` | Int64 | Number of listing photos |
| `street_parked` | Int64 | Whether vehicle is street parked (0/1) |
| `description` | Int64 | Character count of listing description |
| `price_diff` | Float64 | Derived: `actual_price - recommended_price` |

Label: `num_reservations` (Int64, nullable — null for new arrivals).

## Store Architecture

- **Offline store** (Parquet): fleet vehicles only (those with observed reservations). Read by training and serving's `OfflineFeatureReader`.
- **Online store** (Redis): new arrivals only (no observed reservations). Written real-time by `FeatureMaterializer` and batch-backfilled by the pipeline.
- **SQLite**: mutable source of truth (tables: `vehicles`, `reservations`). Seeded from CSVs, augmented by the serving API.

## File Layout

```
features/
  seed.py                           # Seed SQLite from CSVs (idempotent)
  pipeline.py                       # Compute features, write Parquet + Redis
  feature_repo/
    definitions.py                  # Feast Entity, FeatureView, source definitions
    feature_store.yaml              # Feast config (uses $FEAST_REGISTRY, $FEAST_REDIS)
  tests/
    test_features.py                # Unit tests
  pyproject.toml
  uv.lock
  ty.toml
```

## Dependencies

feast[redis], pandas. Dev: pytest.

## Run Locally

```bash
# Set Feast env vars
export FEAST_REGISTRY=../feast-data/registry.db
export FEAST_REDIS=localhost:6379

# Seed the database
cd features && uv run python seed.py \
    --data-dir ../data \
    --db ../feast-data/vehicles.db

# Compute and materialize features
cd features && uv run python pipeline.py \
    --db ../feast-data/vehicles.db \
    --feast-repo feature_repo \
    --parquet-path ../feast-data/vehicle_features.parquet
```

## Tests

```bash
cd features && uv run --group dev pytest tests/ -q
```

## Standards

- Format/lint: ruff (root `ruff.toml`)
- Type check: `uvx ty check --python features/.venv features/`
