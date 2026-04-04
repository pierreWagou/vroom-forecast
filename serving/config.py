"""Serving configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_uri: str = "http://localhost:5001"
    model_name: str = "vroom-forecast"
    host: str = "0.0.0.0"
    port: int = 8000
    feast_repo: str | None = None
    redis_url: str | None = None  # Redis URL for model reload pub/sub

    model_config = {"env_prefix": "SERVING_"}


settings = Settings()
