"""Serving configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_uri: str = "http://localhost:5001"
    model_name: str = "vroom-forecast"
    host: str = "0.0.0.0"
    port: int = 8000
    feast_repo: str | None = None
    redis_url: str | None = None  # Redis URL for model reload pub/sub
    db_path: str = "/feast-data/vehicles.db"  # SQLite path for vehicle persistence
    offline_store_path: str | None = None  # Parquet path for offline feature store
    airflow_url: str | None = None  # Airflow REST API base URL

    model_config = {"env_prefix": "SERVING_"}


settings = Settings()
