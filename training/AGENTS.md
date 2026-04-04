# Training — Agent Instructions

Read README.md for full context.

You are a data scientist writing production ML training code.

- Feature definitions live in `features/feature_repo/definitions.py` — that is the source of truth
- Training reads pre-computed features from the offline store (Parquet) when `--feature-store` is provided
- Falls back to raw CSV loading with local feature computation when `--data-dir` is used
- Every training run MUST be reproducible via MLflow (params, metrics, artifacts)
- Always set the `candidate` alias after registering a new model version
- Use 5-fold cross-validation; log both per-fold and aggregate metrics
- Type-checked with ty; formatted with ruff
