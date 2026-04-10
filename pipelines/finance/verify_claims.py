#!/usr/bin/env python3
"""
verify_claims.py
Cross-references structured claim attributes against XBRL time-series data
to verify whether each claim's stated direction is supported by the numbers.
 
Flask entry point: run_claim_verification_pipeline(claim_records, timeseries_records)
"""
 
import re
import pandas as pd
 
 
# ── Metric mapping ────────────────────────────────────────────────────────────
 
METRIC_COL = {
    "revenue":    "revenue",
    "profit":     "net_profit",
    "ebitda":     "operating_profit",
    "margin":     "profit_margin",
    "expenses":   "total_expenses",
    # proxy mappings
    "subscriber": "revenue",
    "order":      "revenue",
    "deposit":    "revenue",
    "loan":       "revenue",
    "volume":     "revenue",
    "arpu":       "revenue",
    "return":     "profit_margin",
    "provision":  "net_profit",
}
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _normalise_quarter(q: str) -> str:
    """Cleans quarter strings and converts Calendar Years to Indian FY."""
    q = str(q).strip().upper()
    year, quarter = None, None

    m = re.search(r"([1-4])[Qq].*?(?:FY|20)?(\d{2,4})", q)
    if not m: m = re.search(r"[Qq]([1-4]).*?(?:FY|20)?(\d{2,4})", q)
    if not m:
        m = re.match(r"^\d{4}-Q[1-4]$", q)
        if m: year, quarter = int(q[:4]), int(q[-1])

    if m and not year:
        y_val = int(m.group(2))
        year = y_val + 2000 if y_val < 100 else y_val
        quarter = int(m.group(1))

    if year and quarter:
        if "FY" in q: return f"{year}-Q{quarter}"
        # Indian FY Math: Apr-Dec (Q1, Q2, Q3) -> Next Year. Jan-Mar (Q4) -> Same Year.
        fy_year = year + 1 if quarter in [1, 2, 3] else year
        return f"{fy_year}-Q{quarter}"

    return q
 
def _infer_ref_type(sentence: str) -> str:
    t = str(sentence).lower()
    if "yoy" in t or "year" in t:
        return "yoy"
    return "qoq"
 
 
def _evaluate(direction: str, change: float) -> tuple[str, str]:
    if direction == "increase":
        if change > 2:
            return "VERIFIED",      "VERIFIED_STRONG"
        if change < -2:
            return "NOT VERIFIED",  "CONTRADICTS_DIRECTION"
        return "PARTIAL",           "PARTIAL_NO_CHANGE"
 
    if direction == "decrease":
        if change < -2:
            return "VERIFIED",      "VERIFIED_STRONG"
        if change > 2:
            return "NOT VERIFIED",  "CONTRADICTS_DIRECTION"
        return "PARTIAL",           "PARTIAL_NO_CHANGE"
 
    return "PARTIAL", "UNKNOWN_DIRECTION"
 
 
def _get_nearest_row(ts_df: pd.DataFrame, company_key: str, quarter: str):
    """
    Returns (row, match_type). Tries exact match first, then falls back
    to the previous quarter, then the latest available.
    """
    company_rows = ts_df[ts_df["_company_key"] == company_key]
 
    if company_rows.empty:
        return None, "NO_COMPANY_DATA"
 
    exact = company_rows[company_rows["quarter"] == quarter]
    if not exact.empty:
        return exact.iloc[0], "EXACT_MATCH"
 
    sorted_rows = company_rows.sort_values("quarter")
    prev = sorted_rows[sorted_rows["quarter"] <= quarter]
    if not prev.empty:
        return prev.iloc[-1], "FALLBACK_PREV_QUARTER"
 
    return sorted_rows.iloc[-1], "FALLBACK_LATEST"
 
 
# ── Flask entry point ─────────────────────────────────────────────────────────
 
def run_claim_verification_pipeline(
    claim_records: list[dict],
    timeseries_records: list[dict],
) -> list[dict]:
    """
    Main entry point for the Flask app.
 
    Input : claim_records       — from run_attribute_extraction_pipeline()
                                  Keys: company, quarter, sentence, metric,
                                        direction, magnitude, direction_missing
            timeseries_records  — from run_timeseries_pipeline()
                                  Keys: company, quarter, revenue, net_profit, ...
    Output: list[dict] — one record per claim with verification result appended.
            Added keys: actual_change, result, reason
    """
    if not claim_records or not timeseries_records:
        return []
 
    claims = pd.DataFrame(claim_records)
    ts     = pd.DataFrame(timeseries_records)
 
    claims["quarter"]      = claims["quarter"].apply(_normalise_quarter)
    ts["quarter"]          = ts["quarter"].apply(_normalise_quarter)
    ts["_company_key"]     = ts["company"].str.lower().str.strip()
    claims["_company_key"] = claims["company"].str.lower().str.strip()
 
    results = []
 
    for _, row in claims.iterrows():
        company_key = row["_company_key"]
        quarter     = row["quarter"]
        metric      = row["metric"]
        direction   = str(row.get("direction", "")).lower()
        sentence    = row.get("sentence", "")
        base_col    = METRIC_COL.get(metric)
 
        # Unknown metric
        if not base_col or base_col not in ts.columns:
            results.append({
                **row.drop("_company_key").to_dict(),
                "actual_change": None,
                "result":        "SKIPPED",
                "reason":        "MISSING_METRIC_MAPPING",
            })
            continue
 
        ts_row, match_type = _get_nearest_row(ts, company_key, quarter)
 
        if ts_row is None:
            results.append({
                **row.drop("_company_key").to_dict(),
                "actual_change": None,
                "result":        "SKIPPED",
                "reason":        "MISSING_TS_ROW",
            })
            continue
 
        # Resolve change column (YoY or QoQ, with QoQ fallback)
        ref_type     = _infer_ref_type(sentence)
        primary_col  = f"{base_col}_{ref_type}_change"
        fallback_col = f"{base_col}_qoq_change"
 
        actual_change = ts_row.get(primary_col)
        fallback_used = False
 
        if pd.isna(actual_change):
            actual_change = ts_row.get(fallback_col)
            fallback_used = True
 
        if pd.isna(actual_change):
            results.append({
                **row.drop("_company_key").to_dict(),
                "actual_change": None,
                "result":        "SKIPPED",
                "reason":        "MISSING_VALUE",
            })
            continue
 
        result, reason = _evaluate(direction, float(actual_change))
        tag = match_type + ("|FALLBACK_CHANGE" if fallback_used else "")
 
        results.append({
            **row.drop("_company_key").to_dict(),
            "actual_change": round(float(actual_change), 2),
            "result":        result,
            "reason":        f"{reason} | {tag}",
        })
 
    return results