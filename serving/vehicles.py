"""Vehicle persistence — SQLite storage + event emission on save."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from serving.config import settings
from serving.schemas import VehicleFeatures, VehicleRecord

logger = logging.getLogger(__name__)

DB_PATH = "/feast-data/vehicles.db"
VEHICLE_SAVED_CHANNEL = "vroom-forecast:vehicle-saved"


def _get_db() -> sqlite3.Connection:
    """Get a connection to the vehicle database, creating the table if needed."""
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            technology INTEGER NOT NULL,
            actual_price REAL NOT NULL,
            recommended_price REAL NOT NULL,
            num_images INTEGER NOT NULL,
            street_parked INTEGER NOT NULL,
            description INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    return conn


def save_vehicle(vehicle: VehicleFeatures) -> int:
    """Save a vehicle's raw attributes to SQLite and emit an event.

    Returns the assigned vehicle_id. A feature worker listening on Redis
    pub/sub will compute and materialize features to the online store.
    """
    conn = _get_db()
    cursor = conn.execute(
        """
        INSERT INTO vehicles (technology, actual_price, recommended_price,
                              num_images, street_parked, description, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            vehicle.technology,
            vehicle.actual_price,
            vehicle.recommended_price,
            vehicle.num_images,
            vehicle.street_parked,
            vehicle.description,
            datetime.now(tz=UTC).isoformat(),
        ),
    )
    conn.commit()
    vehicle_id = cursor.lastrowid
    assert vehicle_id is not None
    conn.close()

    logger.info("Saved vehicle %d to database.", vehicle_id)
    _emit_vehicle_saved(vehicle_id, vehicle)
    return vehicle_id


def _emit_vehicle_saved(vehicle_id: int, vehicle: VehicleFeatures) -> None:
    """Publish a vehicle-saved event to Redis pub/sub."""
    if settings.redis_url is None:
        logger.info("No Redis URL configured — skipping event emission.")
        return

    import redis

    try:
        r = redis.from_url(settings.redis_url)
        payload = json.dumps(
            {
                "vehicle_id": vehicle_id,
                "technology": vehicle.technology,
                "actual_price": vehicle.actual_price,
                "recommended_price": vehicle.recommended_price,
                "num_images": vehicle.num_images,
                "street_parked": vehicle.street_parked,
                "description": vehicle.description,
            }
        )
        listeners = r.publish(VEHICLE_SAVED_CHANNEL, payload)
        logger.info(
            "Published vehicle-saved event for #%d (%d listeners).",
            vehicle_id,
            listeners,
        )
    except Exception:
        logger.exception("Failed to publish vehicle-saved event for #%d.", vehicle_id)


def list_vehicles() -> list[VehicleRecord]:
    """List all saved vehicles."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT vehicle_id, technology, actual_price, recommended_price, "
        "num_images, street_parked, description FROM vehicles ORDER BY vehicle_id"
    ).fetchall()
    conn.close()
    return [
        VehicleRecord(
            vehicle_id=r[0],
            technology=r[1],
            actual_price=r[2],
            recommended_price=r[3],
            num_images=r[4],
            street_parked=r[5],
            description=r[6],
        )
        for r in rows
    ]
