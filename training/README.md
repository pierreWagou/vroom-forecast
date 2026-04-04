# Training Pipeline

Independent uv project that trains a RandomForest model to predict vehicle
reservation counts from listing attributes.

## Data Flow

```mermaid
graph LR
    PQ[Parquet<br/>offline store] -->|--feature-store| Train
    CSV[data/*.csv] -->|--data-dir<br/>fallback| Train
    Train -->|log metrics + model| MLflow[(MLflow)]
    Train -->|set candidate alias| MLflow
```

## Running

```bash
# From the offline feature store (preferred — uses materialized features):
uv run --project training python -m training \
    --feature-store /feast-data/vehicle_features.parquet \
    --mlflow-uri http://localhost:5001

# From raw CSVs (fallback — computes features locally):
uv run --project training python -m training \
    --data-dir data \
    --mlflow-uri http://localhost:5001
```

## What it does

1. Loads features from the offline store (Parquet) or raw CSVs (fallback)
2. Trains a `RandomForestRegressor` (200 trees, max_depth=10) with 5-fold CV
3. Logs params, metrics, and model artifact to MLflow
4. Registers the model version and sets the `candidate` alias

## Key files

- `train.py` — Pipeline logic (load, train, evaluate, register)
- `__main__.py` — CLI entry point
- `pyproject.toml` — Dependencies: pandas, scikit-learn, mlflow, numpy, pyarrow

## Feature columns

`technology`, `actual_price`, `recommended_price`, `num_images`,
`street_parked`, `description`, `price_diff`, `price_ratio`
