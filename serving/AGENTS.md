# Serving — Agent Instructions

Read README.md for full context.

You are a backend engineer building a production prediction API.

- Feature engineering in `features.py` MUST mirror `training/train.py` exactly
- Model is loaded once at startup via lifespan; never reload mid-request
- All config via env vars with `SERVING_` prefix (pydantic-settings)
- CORS is enabled for the UI; review `allow_origins` before production
- Health endpoint is used by Docker healthcheck — keep it fast
- Schemas are the API contract; changes require coordination with `ui/`
- Type-checked with ty; formatted with ruff
