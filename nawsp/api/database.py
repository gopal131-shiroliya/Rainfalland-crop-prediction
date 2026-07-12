import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Vercel Functions are read-only except for /tmp. Local development keeps the
# database beside the application; deployed demo history is intentionally
# temporary and can reset when Vercel creates a new function instance.
DB_PATH = Path("/tmp/predictions.db") if os.getenv("VERCEL") else BASE_DIR / "predictions.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cursor, table_name, column_name, definition):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            N REAL,
            P REAL,
            K REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            wind_speed REAL,
            rainfall_prediction REAL,
            recommended_crop TEXT
        )
        """
    )

    # Lightweight migration for older DBs.
    _ensure_column(cursor, "predictions", "created_at", "TEXT")
    _ensure_column(cursor, "predictions", "N", "REAL")
    _ensure_column(cursor, "predictions", "P", "REAL")
    _ensure_column(cursor, "predictions", "K", "REAL")
    _ensure_column(cursor, "predictions", "ph", "REAL")

    cursor.execute(
        """
        UPDATE predictions
        SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)
        WHERE created_at IS NULL OR TRIM(created_at) = ''
        """
    )

    conn.commit()
    conn.close()
