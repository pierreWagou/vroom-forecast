# Feature Store

Feast-based feature store with offline (Parquet) and online (Redis) stores.
Single source of truth for all vehicle features.

## Running

```bash
# Run the feature pipeline (compute features + materialize to offline + online):
uv run --project features python -m features \
    --data-dir data \
    --feast-repo features/feature_repo

# In Airflow (via the training DAG's materialize task):
# This runs automatically before training.
```

## Architecture

```
data/vehicles.csv + reservations.csv
        │
        ▼
  Feature Pipeline (pipeline.py)
        │
        ├──▶ Parquet (offline store)  ──▶ Training (get_historical_features)
        │
        └──▶ Redis (online store)     ──▶ Serving API (get_online_features)
```

## Key files

- `feature_repo/feature_store.yaml` — Feast config (file offline + Redis online)
- `feature_repo/definitions.py` — Entity, FeatureView, schema, feature refs
- `pipeline.py` — Loads raw data, computes features, writes Parquet, materializes
- `__main__.py` — CLI entry point
- `pyproject.toml` — Dependencies: feast[redis], pandas, numpy

## Feature columns

| Feature | Type | Source |
|---------|------|--------|
| technology | Int64 | Raw |
| actual_price | Float64 | Raw |
| recommended_price | Float64 | Raw |
| num_images | Int64 | Raw |
| street_parked | Int64 | Raw |
| description | Int64 | Raw |
| price_diff | Float64 | Derived (actual - recommended) |
| price_ratio | Float64 | Derived (actual / recommended) |
| num_reservations | Int64 | Aggregated (label, not a feature) |
