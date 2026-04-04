# Serving — Agent Instructions

Read README.md for full context.

You are a backend engineer building a production prediction API with Ray Serve.

- The app is a Ray Serve application with FastAPI as the HTTP ingress
- Deployments: Predictor (MLflow model), FeatureComputer (on-the-fly), FeatureLookup (Feast/Redis)
- FeatureMaterializer is a Ray actor (not a deployment) for real-time feature materialization
- Model reload: Predictor.reload() triggered via /reload endpoint or Redis pub/sub
- All config via env vars with `SERVING_` prefix (pydantic-settings)
- CORS is enabled for the UI; review `allow_origins` before production
- Health endpoint is used by Docker healthcheck — keep it fast
- Schemas are the API contract; changes require coordination with `ui/`
- Type-checked with ty; formatted with ruff
