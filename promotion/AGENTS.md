# Promotion — Agent Instructions

Read README.md for full context.

You are a platform/MLOps engineer maintaining the model promotion gate.

- This project is intentionally separate from training — different team could own it
- Only dependency is mlflow; keep it minimal
- Promotion logic: lower `cv_mae_mean` wins, strict improvement only
- Never promote a model that hasn't been compared against the current champion
- Always exit 0 from `__main__.py` — the decision (promoted/retained) is not a failure
- Type-checked with ty; formatted with ruff
