"""
Seed the SQLite database from CSVs.

Loads vehicles.csv and reservations.csv into the shared SQLite database.
Idempotent — skips if data already exists.

Usage:
    uv run python seed.py --data-dir /path/to/data --db /feast-data/vehicles.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "/feast-data/vehicles.db"


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the database and tables if they don't exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads/writes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            technology INTEGER NOT NULL,
            actual_price REAL NOT NULL,
            recommended_price REAL NOT NULL,
            num_images INTEGER NOT NULL,
            street_parked INTEGER NOT NULL,
            description INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'csv'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
        )
    """)
    conn.commit()
    return conn


def seed(data_dir: Path, db_path: str) -> None:
    """Seed the database from CSVs. Skips if already seeded."""
    conn = init_db(db_path)

    # Check if already seeded
    count = conn.execute("SELECT COUNT(*) FROM vehicles WHERE source = 'csv'").fetchone()[0]
    if count > 0:
        logger.info("Database already seeded with %d CSV vehicles — skipping.", count)
        conn.close()
        return

    # Load and insert vehicles
    vehicles = pd.read_csv(data_dir / "vehicles.csv")
    vehicles["source"] = "csv"
    vehicles.to_sql("vehicles", conn, if_exists="append", index=False)
    logger.info("Seeded %d vehicles from CSV.", len(vehicles))

    # Load and insert reservations
    reservations = pd.read_csv(data_dir / "reservations.csv")
    reservations.to_sql("reservations", conn, if_exists="append", index=False)
    logger.info("Seeded %d reservations from CSV.", len(reservations))

    conn.commit()
    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the database from CSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=str, default=DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed(data_dir=args.data_dir, db_path=args.db)
