"""
Vroom Forecast — Feast Feature Definitions

Single source of truth for all vehicle features. Both training (offline)
and serving (online/Redis) read from these definitions.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

# ── Entity ───────────────────────────────────────────────────────────────────

vehicle = Entity(
    name="vehicle",
    join_keys=["vehicle_id"],
    value_type=ValueType.INT64,
    description="A vehicle listed on the platform",
)

# ── Data source (offline — Parquet file written by the feature pipeline) ─────

vehicle_features_source = FileSource(
    name="vehicle_features_source",
    path="/feast-data/vehicle_features.parquet",
    timestamp_field="event_timestamp",
)

# ── Feature view ─────────────────────────────────────────────────────────────

vehicle_features_view = FeatureView(
    name="vehicle_features",
    entities=[vehicle],
    schema=[
        # Raw attributes (model features)
        Field(name="technology", dtype=Int64),
        Field(name="num_images", dtype=Int64),
        Field(name="street_parked", dtype=Int64),
        Field(name="description", dtype=Int64),
        # Derived feature (computed by the feature pipeline)
        Field(name="price_diff", dtype=Float64),
        # Label (stored for training convenience, not used as a feature)
        Field(name="num_reservations", dtype=Int64),
    ],
    source=vehicle_features_source,
    online=True,
    ttl=timedelta(days=365),
)

# ── Feature list for model training / serving ────────────────────────────────
# This is the canonical list — training and serving must use the same features.

FEATURE_REFS = [
    "vehicle_features:technology",
    "vehicle_features:num_images",
    "vehicle_features:street_parked",
    "vehicle_features:description",
    "vehicle_features:price_diff",
]

LABEL_REF = "vehicle_features:num_reservations"
