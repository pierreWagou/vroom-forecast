"""Serving configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_uri: str = "http://localhost:5001"
    model_name: str = "vroom-forecast"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "SERVING_"}


settings = Settings()
