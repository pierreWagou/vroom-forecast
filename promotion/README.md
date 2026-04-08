# Promotion Pipeline

Independent uv project that compares a candidate model against the current
champion and promotes it if the candidate has a lower `cv_mae_mean`.

On successful promotion, publishes an event to Redis pub/sub so downstream
consumers (e.g. the Ray Serve Predictor) can reload the model.

## Running

```bash
# Resolve candidate from MLflow "candidate" alias:
uv run --project promotion python -m promotion --mlflow-uri http://localhost:5001

# With explicit version + Redis notification:
uv run --project promotion python -m promotion \
    --version 5 \
    --mlflow-uri http://localhost:5001 \
    --redis-url redis://localhost:6379
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--version` | *(none)* | Explicit model version (otherwise resolves from `--candidate-alias`) |
| `--candidate-alias` | `candidate` | MLflow alias to resolve candidate from |
| `--mlflow-uri` | `http://localhost:5001` | MLflow tracking server URI |
| `--model-name` | `vroom-forecast` | Registered model name |
| `--metric` | `cv_mae_mean` | Metric to compare (lower is better) |
| `--redis-url` | *(none)* | Redis URL for promotion notification |

## What it does

1. Resolves the candidate version (by explicit `--version` or `candidate` alias)
2. Fetches the candidate's `cv_mae_mean` metric from MLflow
3. Fetches the current champion's `cv_mae_mean` (if one exists)
4. If candidate < champion: promotes candidate to `champion` alias
5. If no champion exists: promotes candidate as first champion
6. On promotion: publishes event to Redis channel `vroom-forecast:model-promoted`

## Key files

- `promote.py` — Promotion logic, Redis notification, CLI arg parsing
- `__main__.py` — CLI entry point (exits 0 on success; exits 1 on exception)
- `pyproject.toml` — Dependencies: mlflow, redis
