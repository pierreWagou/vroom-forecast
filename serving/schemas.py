"""Request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class VehicleFeatures(BaseModel):
    """Raw vehicle attributes (before feature engineering)."""

    technology: int = Field(..., ge=0, le=1, description="0=none, 1=installed")
    actual_price: float = Field(..., gt=0, description="Daily price set by owner")
    recommended_price: float = Field(..., gt=0, description="Market price")
    num_images: int = Field(..., ge=0, description="Number of photos")
    street_parked: int = Field(..., ge=0, le=1, description="0=no, 1=yes")
    description: int = Field(..., ge=0, description="Character count of description")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "technology": 1,
                    "actual_price": 45.0,
                    "recommended_price": 50.0,
                    "num_images": 8,
                    "street_parked": 0,
                    "description": 250,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_reservations: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class BenchmarkRequest(BaseModel):
    n_iterations: int = Field(default=1000, ge=1, le=100_000)
    vehicle: VehicleFeatures


class BenchmarkByIdRequest(BaseModel):
    n_iterations: int = Field(default=1000, ge=1, le=100_000)
    vehicle_id: int = Field(..., gt=0)


class BenchmarkResponse(BaseModel):
    n_iterations: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_version: str
    source: str = "raw"  # "raw" or "online_store"
    # Step-level breakdown
    avg_features_ms: float = 0.0  # feature computation or lookup time
    avg_predict_ms: float = 0.0  # model.predict() time


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    mlflow_uri: str
    feast_online: bool


class VehicleIdRequest(BaseModel):
    """Predict by vehicle ID — features are looked up from the online store (Redis)."""

    vehicle_id: int = Field(..., gt=0, description="Vehicle ID to look up in the feature store")


class ReloadResponse(BaseModel):
    status: str
    previous_version: str
    current_version: str


class SaveVehicleResponse(BaseModel):
    vehicle_id: int
    status: str
    event_published: bool = True  # False if Redis was unavailable


class VehicleRecord(BaseModel):
    """A stored vehicle with its ID, attributes, source, and reservation count.

    num_reservations is None for new arrivals (no observation yet) and an int
    (including 0) for vehicles with observed outcomes.
    """

    vehicle_id: int
    technology: int
    actual_price: float
    recommended_price: float
    num_images: int
    street_parked: int
    description: int
    source: str = "csv"  # "csv" or "ui"
    num_reservations: int | None = None


class ComputedFeatures(BaseModel):
    """Features as computed by the feature pipeline and stored in the feature store.

    The 5 model features: technology, num_images, street_parked, description,
    price_diff. Raw prices are vehicle attributes, not model features.
    """

    vehicle_id: int
    technology: int | None = None
    num_images: int | None = None
    street_parked: int | None = None
    description: int | None = None
    price_diff: float | None = None
    materialized: bool = False
    store: Literal["offline", "online", "none"] = "none"


class StoreDetails(BaseModel):
    """Operational info about a single store (offline or online)."""

    available: bool = False
    type: str = ""
    path: str | None = None
    size_bytes: int | None = None
    last_modified: float | None = None
    used_memory_human: str | None = None
    keys: int | None = None
    redis_url: str | None = None


class FeatureViewInfo(BaseModel):
    """Feature view definition summary."""

    name: str = "vehicle_features"
    entity: str = "vehicle"
    entity_key: str = "vehicle_id"
    features: list[str] = ["technology", "num_images", "street_parked", "description", "price_diff"]
    label: str = "num_reservations"
    ttl_days: int | None = 365


class StoreInfoResponse(BaseModel):
    """Combined offline + online store info."""

    offline_store: StoreDetails
    online_store: StoreDetails
    feature_view: FeatureViewInfo = FeatureViewInfo()


class DeleteVehicleResponse(BaseModel):
    """Response for vehicle deletion."""

    status: str
    vehicle_id: int


class TriggerDagResponse(BaseModel):
    """Response for triggering an Airflow DAG."""

    status: str
    dag_id: str
    dag_run_id: str | None = None


class DagRunStatus(BaseModel):
    """Status of an Airflow DAG run."""

    dag_id: str
    dag_run_id: str
    state: str  # queued, running, success, failed
