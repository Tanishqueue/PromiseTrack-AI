"""
db.py
SQLite database setup. Single source of truth for schema and connection.
All tables are created here on first run — import get_db() everywhere else.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

import config

DB_PATH = Path(config.BASE_DIR) / "promisetrack.db"


def init_db() -> None:
    """Create all tables if they don't exist. Call once in create_app()."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS companies (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_name  TEXT    UNIQUE NOT NULL,
            display_name TEXT    NOT NULL,
            status       TEXT    NOT NULL DEFAULT 'pending',
            processed_at TIMESTAMP,
            error_msg    TEXT
        );

        CREATE TABLE IF NOT EXISTS analysis_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id  INTEGER NOT NULL REFERENCES companies(id),
            mode        TEXT    NOT NULL,
            result_json TEXT    NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(company_id, mode)
        );

        CREATE TABLE IF NOT EXISTS claims (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id    INTEGER NOT NULL REFERENCES companies(id),
            quarter       TEXT,
            sentence      TEXT,
            metric        TEXT,
            direction     TEXT,
            magnitude     TEXT,
            result        TEXT,
            actual_change REAL,
            confidence    REAL
        );

        CREATE TABLE IF NOT EXISTS timeseries (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id                  INTEGER NOT NULL REFERENCES companies(id),
            quarter                     TEXT,
            revenue                     REAL,
            net_profit                  REAL,
            operating_profit            REAL,
            profit_margin               REAL,
            revenue_qoq_change          REAL,
            net_profit_qoq_change       REAL,
            operating_profit_qoq_change REAL,
            profit_margin_qoq_change    REAL,
            revenue_yoy_change          REAL,
            net_profit_yoy_change       REAL,
            operating_profit_yoy_change REAL,
            profit_margin_yoy_change    REAL,
            UNIQUE(company_id, quarter)
        );

        CREATE TABLE IF NOT EXISTS risk (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id              INTEGER NOT NULL REFERENCES companies(id),
            quarter                 TEXT,
            total_claims            INTEGER,
            verification_rate       REAL,
            failure_rate            REAL,
            partial_rate            REAL,
            direction_mismatch_rate REAL,
            consistency_score       REAL,
            risk_drift              REAL,
            warning_flag            INTEGER,
            UNIQUE(company_id, quarter)
        );
        """)


@contextmanager
def get_db():
    """Context manager that yields a SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()