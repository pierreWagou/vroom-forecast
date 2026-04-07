"""Vehicle persistence — SQLite storage + event emission on save."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from serving.config import settings
from serving.model import VEHICLE_SAVED_CHANNEL
from serving.schemas import VehicleFeatures, VehicleRecord

logger = logging.getLogger(__name__)

DB_PATH = settings.db_path


def _get_db() -> sqlite3.Connection:
    """Get a connection to the shared vehicle database."""
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads/writes
    # Table is created by features/seed.py — but ensure it exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            technology INTEGER NOT NULL,
            actual_price REAL NOT NULL,
            recommended_price REAL NOT NULL,
            num_images INTEGER NOT NULL,
            street_parked INTEGER NOT NULL,
            description INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'ui'
        )
    """)
    # Migrate: add source column if missing (old databases)
    try:
        conn.execute("SELECT source FROM vehicles LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE vehicles ADD COLUMN source TEXT NOT NULL DEFAULT 'csv'")
        conn.commit()
    return conn


def save_vehicle(vehicle: VehicleFeatures) -> tuple[int, bool]:
    """Save a vehicle to the shared database and emit an event.

    Returns (vehicle_id, event_published). The FeatureMaterializer Ray actor
    will compute and materialize features to the online store.
    """
    conn = _get_db()
    try:
        cursor = conn.execute(
            """
            INSERT INTO vehicles (technology, actual_price, recommended_price,
                                  num_images, street_parked, description, source)
            VALUES (?, ?, ?, ?, ?, ?, 'ui')
            """,
            (
                vehicle.technology,
                vehicle.actual_price,
                vehicle.recommended_price,
                vehicle.num_images,
                vehicle.street_parked,
                vehicle.description,
            ),
        )
        conn.commit()
        vehicle_id = cursor.lastrowid
        if vehicle_id is None:
            raise RuntimeError("Failed to insert vehicle — no lastrowid returned")
    finally:
        conn.close()

    logger.info("Saved vehicle %d to database.", vehicle_id)
    event_published = _emit_vehicle_saved(vehicle_id, vehicle)
    return vehicle_id, event_published


def _emit_vehicle_saved(vehicle_id: int, vehicle: VehicleFeatures) -> bool:
    """Publish a vehicle-saved event to Redis pub/sub. Returns True if published."""
    if settings.redis_url is None:
        logger.info("No Redis URL configured — skipping event emission.")
        return False

    import redis

    try:
        r = redis.from_url(settings.redis_url)
        try:
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
            return True
        finally:
            r.close()
    except Exception:
        logger.exception("Failed to publish vehicle-saved event for #%d.", vehicle_id)
        return False


def delete_vehicle(vehicle_id: int) -> bool:
    """Delete a new arrival vehicle. Only allows deleting ui-sourced vehicles.

    Returns True if the vehicle was deleted, False if not found or not deletable.
    """
    conn = _get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM vehicles WHERE vehicle_id = ? AND source = 'ui'",
            (vehicle_id,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
    finally:
        conn.close()

    if deleted:
        logger.info("Deleted vehicle %d from database.", vehicle_id)
    else:
        logger.warning("Vehicle %d not found or not deletable (source != 'ui').", vehicle_id)
    return deleted


def list_vehicles() -> list[VehicleRecord]:
    """List all vehicles with reservation counts and source."""
    conn = _get_db()

    try:
        # Check if reservations table exists (may not if seed hasn't run yet)
        has_reservations = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='reservations'"
        ).fetchone()[0]

        if has_reservations:
            rows = conn.execute(
                """
                SELECT v.vehicle_id, v.technology, v.actual_price, v.recommended_price,
                       v.num_images, v.street_parked, v.description, v.source,
                       CASE
                           WHEN v.source = 'csv' THEN COALESCE(r.cnt, 0)
                           ELSE r.cnt
                       END as num_reservations
                FROM vehicles v
                LEFT JOIN (
                    SELECT vehicle_id, COUNT(*) as cnt FROM reservations GROUP BY vehicle_id
                ) r ON v.vehicle_id = r.vehicle_id
                ORDER BY v.vehicle_id
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT vehicle_id, technology, actual_price, recommended_price,
                       num_images, street_parked, description, source,
                       CASE WHEN source = 'csv' THEN 0 ELSE NULL END as num_reservations
                FROM vehicles ORDER BY vehicle_id
                """
            ).fetchall()
    finally:
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
            source=r[7],
            num_reservations=r[8],
        )
        for r in rows
    ]
