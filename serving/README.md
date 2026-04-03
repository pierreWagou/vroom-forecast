# Serving API

Independent uv project — FastAPI application that serves reservation count
predictions from the champion model in MLflow.

## Running

```bash
# Local (expects MLflow at localhost:5001):
uv run --project serving python -m serving

# With custom MLflow URI:
SERVING_MLFLOW_URI=http://localhost:5001 uv run --project serving python -m serving
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model info |
| POST | `/predict` | Single vehicle prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000) |
| POST | `/benchmark` | Run N predictions, report latency stats |

Interactive docs at `http://localhost:8000/docs`.

## Architecture

```
app.py        — FastAPI app, routes, CORS, lifespan
config.py     — Settings from env vars (SERVING_MLFLOW_URI, etc.)
schemas.py    — Pydantic request/response models
features.py   — Feature engineering (mirrors training logic)
model.py      — Champion model loading + inference
__main__.py   — Entry point (uvicorn)
Dockerfile    — Standalone container for docker-compose
```

## Configuration

All settings via environment variables with `SERVING_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVING_MLFLOW_URI` | `http://localhost:5001` | MLflow tracking server |
| `SERVING_MODEL_NAME` | `vroom-forecast` | Registered model name |
| `SERVING_HOST` | `0.0.0.0` | Bind address |
| `SERVING_PORT` | `8000` | Bind port |

## Docker

```bash
docker compose up serving    # runs on port 8000
```
