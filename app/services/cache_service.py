"""
app/services/cache_service.py
All SQLite read/write operations for the cache layer.
No pipeline logic lives here — only DB operations.
"""

import json
import re
from datetime import datetime
from typing import Optional

from db import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_display_name(folder_name: str) -> str:
    """Strip year/quarter suffixes from folder names."""
    name = re.sub(r'\s*\d{4}\s*Quarterly\s*Data\s*', '', folder_name, flags=re.IGNORECASE)
    name = re.sub(r'\s*Quarterly\s*Data\s*', '', name, flags=re.IGNORECASE)
    return name.strip()


# ── Company registry ──────────────────────────────────────────────────────────

def get_all_companies() -> list:
    """Return all companies with their processing status."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, folder_name, display_name, status, processed_at FROM companies ORDER BY display_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_ready_companies() -> list:
    """Return only companies that have been successfully processed."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, folder_name, display_name, processed_at FROM companies WHERE status = 'ready' ORDER BY display_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_company_by_folder(folder_name: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM companies WHERE folder_name = ?", (folder_name,)
        ).fetchone()
    return dict(row) if row else None


def upsert_company(folder_name: str, status: str = "pending", error_msg: str = None) -> int:
    """Insert or update a company record. Returns company id."""
    display = _clean_display_name(folder_name)
    with get_db() as conn:
        conn.execute("""
            INSERT INTO companies (folder_name, display_name, status, processed_at, error_msg)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(folder_name) DO UPDATE SET
                status       = excluded.status,
                processed_at = excluded.processed_at,
                error_msg    = excluded.error_msg
        """, (folder_name, display, status,
              datetime.utcnow().isoformat() if status == "ready" else None,
              error_msg))
        row = conn.execute(
            "SELECT id FROM companies WHERE folder_name = ?", (folder_name,)
        ).fetchone()
    return row["id"]


# ── Analysis cache ────────────────────────────────────────────────────────────

def get_cached_analysis(company_id: int, mode: str) -> Optional[dict]:
    """Return cached analysis result or None if not found."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT result_json FROM analysis_cache WHERE company_id = ? AND mode = ?",
            (company_id, mode)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def save_analysis(company_id: int, mode: str, result: dict) -> None:
    """Save (overwrite) analysis result for a company+mode."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO analysis_cache (company_id, mode, result_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(company_id, mode) DO UPDATE SET
                result_json = excluded.result_json,
                created_at  = excluded.created_at
        """, (company_id, mode, json.dumps(result), datetime.utcnow().isoformat()))


# ── Claims ────────────────────────────────────────────────────────────────────

def save_claims(company_id: int, claims: list) -> None:
    """Overwrite all claims for a company."""
    with get_db() as conn:
        conn.execute("DELETE FROM claims WHERE company_id = ?", (company_id,))
        conn.executemany("""
            INSERT INTO claims
              (company_id, quarter, sentence, metric, direction, magnitude, result, actual_change, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            company_id,
            c.get("quarter"),
            c.get("sentence"),
            c.get("metric"),
            c.get("direction"),
            c.get("magnitude"),
            c.get("result"),
            c.get("actual_change"),
            c.get("confidence"),
        ) for c in claims])


def get_claims(company_id: int) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE company_id = ? ORDER BY quarter", (company_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Timeseries ────────────────────────────────────────────────────────────────

def save_timeseries(company_id: int, ts_records: list) -> None:
    """Overwrite all timeseries rows for a company."""
    with get_db() as conn:
        conn.execute("DELETE FROM timeseries WHERE company_id = ?", (company_id,))
        conn.executemany("""
            INSERT OR REPLACE INTO timeseries
              (company_id, quarter, revenue, net_profit, operating_profit, profit_margin,
               revenue_qoq_change, net_profit_qoq_change, operating_profit_qoq_change, profit_margin_qoq_change,
               revenue_yoy_change, net_profit_yoy_change, operating_profit_yoy_change, profit_margin_yoy_change)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            company_id,
            r.get("quarter"),
            r.get("revenue"),
            r.get("net_profit"),
            r.get("operating_profit"),
            r.get("profit_margin"),
            r.get("revenue_qoq_change"),
            r.get("net_profit_qoq_change"),
            r.get("operating_profit_qoq_change"),
            r.get("profit_margin_qoq_change"),
            r.get("revenue_yoy_change"),
            r.get("net_profit_yoy_change"),
            r.get("operating_profit_yoy_change"),
            r.get("profit_margin_yoy_change"),
        ) for r in ts_records])


# ── Risk ──────────────────────────────────────────────────────────────────────

def save_risk(company_id: int, risk_records: list) -> None:
    """Overwrite all risk rows for a company."""
    with get_db() as conn:
        conn.execute("DELETE FROM risk WHERE company_id = ?", (company_id,))
        conn.executemany("""
            INSERT OR REPLACE INTO risk
              (company_id, quarter, total_claims, verification_rate, failure_rate,
               partial_rate, direction_mismatch_rate, consistency_score, risk_drift, warning_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            company_id,
            r.get("quarter"),
            r.get("total_claims"),
            r.get("verification_rate"),
            r.get("failure_rate"),
            r.get("partial_rate"),
            r.get("direction_mismatch_rate"),
            r.get("consistency_score"),
            r.get("risk_drift"),
            r.get("warning_flag"),
        ) for r in risk_records])