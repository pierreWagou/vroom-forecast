# Promotion Pipeline

Independent uv project that compares a candidate model against the current
champion and promotes it if the candidate has a lower `cv_mae_mean`.

## Running

```bash
# Resolve candidate from MLflow "candidate" alias:
uv run --project promotion python -m promotion --mlflow-uri http://localhost:5001

# Or with explicit version:
uv run --project promotion python -m promotion --version 5 --mlflow-uri http://localhost:5001
```

## What it does

1. Resolves the candidate version (by explicit `--version` or `candidate` alias)
2. Fetches the candidate's `cv_mae_mean` metric from MLflow
3. Fetches the current champion's `cv_mae_mean` (if one exists)
4. If candidate < champion: promotes candidate to `champion` alias
5. If no champion exists: promotes candidate as first champion

## Key files

- `promote.py` — Promotion logic + CLI arg parsing
- `__main__.py` — CLI entry point (always exits 0; prints `promoted` or `retained`)
- `pyproject.toml` — Dependencies: mlflow only
