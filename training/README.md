# Training Pipeline

Independent uv project that trains a RandomForest model to predict vehicle
reservation counts from listing attributes.

## Data Flow

```mermaid
graph LR
    PQ[Parquet<br/>offline store] -->|--feature-store| Train
    Train -->|log metrics + model| MLflow[(MLflow)]
    Train -->|set candidate alias| MLflow
```

## Running

```bash
# From the offline feature store (materialized by the feature pipeline):
uv run --project training python -m training \
    --feature-store feast-data/vehicle_features.parquet \
    --mlflow-uri http://localhost:5001
```

## What it does

1. Loads features from the offline store (Parquet file written by the feature pipeline)
2. Trains a `RandomForestRegressor` (200 trees, max_depth=10) with 5-fold CV
3. Logs params, metrics, and model artifact to MLflow
4. Registers the model version and sets the `candidate` alias

## Key files

- `train.py` — Pipeline logic (load, train, evaluate, register)
- `__main__.py` — CLI entry point
- `pyproject.toml` — Dependencies: pandas, scikit-learn, mlflow, pyarrow

## Feature columns

`technology`, `num_images`, `street_parked`, `description`, `price_diff`

Target: `num_reservations`

> Raw prices (`actual_price`, `recommended_price`) are vehicle attributes used
> to compute `price_diff` but are not model inputs — `price_diff` captures the
> full pricing signal with less collinearity. See [exploration](../exploration/).
