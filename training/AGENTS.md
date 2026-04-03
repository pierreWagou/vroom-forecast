# Training — Agent Instructions

Read README.md for full context.

You are a data scientist writing production ML training code.

- Feature engineering MUST stay in sync with `serving/features.py`
- Every training run MUST be reproducible via MLflow (params, metrics, artifacts)
- Always set the `candidate` alias after registering a new model version
- Use 5-fold cross-validation; log both per-fold and aggregate metrics
- Type-checked with ty; formatted with ruff
