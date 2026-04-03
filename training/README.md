# Training Pipeline

Independent uv project that trains a RandomForest model to predict vehicle
reservation counts from listing attributes.

## Running

```bash
uv run --project training python -m training --data-dir data --mlflow-uri http://localhost:5001
```

## What it does

1. Loads `data/vehicles.csv` + `data/reservations.csv`
2. Aggregates reservation counts per vehicle
3. Engineers features: `price_diff`, `price_ratio`
4. Trains `RandomForestRegressor` (200 trees, max_depth=10) with 5-fold CV
5. Logs params, metrics, and model artifact to MLflow
6. Registers the model version and sets the `candidate` alias

## Key files

- `train.py` — All pipeline logic (load, feature engineering, train, evaluate, register)
- `__main__.py` — CLI entry point
- `pyproject.toml` — Dependencies: pandas, scikit-learn, mlflow, numpy

## Feature columns

`technology`, `actual_price`, `recommended_price`, `num_images`,
`street_parked`, `description`, `price_diff`, `price_ratio`
