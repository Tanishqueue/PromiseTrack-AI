#!/usr/bin/env python3
"""
prepare_timeseries_data.py
Builds a wide time-series DataFrame from raw XBRL numeric records,
adding QoQ / YoY change features and profit margin.
 
Flask entry point: run_timeseries_pipeline(xbrl_records)
"""
 
import numpy as np
import pandas as pd
 
 
# ── Config ────────────────────────────────────────────────────────────────────
 
KEY_METRICS = {
    "revenue":          ["revenue", "total_income"],
    "net_profit":       ["net_profit", "profit_after_tax"],
    "operating_profit": ["operating_profit"],
}
 
_FEATURE_COLS = ["revenue", "net_profit", "operating_profit", "profit_margin"]
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _sort_key(q: str) -> tuple:
    try:
        year, qn = q.split("-Q")
        return (int(year), int(qn))
    except Exception:
        return (0, 0)
 
def _safe_pct_change(series: pd.Series) -> pd.Series:
    return (series - series.shift(1)) / series.shift(1) * 100
 
 
# ── Flask entry point ─────────────────────────────────────────────────────────
 
def run_timeseries_pipeline(xbrl_records: list[dict]) -> list[dict]:
    """
    Main entry point for the Flask app.
 
    Input : list[dict] from run_xbrl_extraction_pipeline()
            Keys: company, quarter, metric, value
    Output: list[dict] — one row per (company, quarter) with derived
            time-series features (QoQ, YoY, margin).
    """
    if not xbrl_records:
        return []
 
    df = pd.DataFrame(xbrl_records)
 
    # ── Consolidate to wide format ────────────────────────────────────────────
    records = []
    for company in df["company"].unique():
        cdf = df[df["company"] == company]
        for quarter in cdf["quarter"].unique():
            qdf = cdf[cdf["quarter"] == quarter]
            row = {"company": company, "quarter": quarter}
            for target, sources in KEY_METRICS.items():
                val = None
                for src in sources:
                    subset = qdf[qdf["metric"] == src]
                    if not subset.empty:
                        # Use max to get the consolidated/total figure,
                        # not a sub-component that may have snuck through.
                        val = subset["value"].max()
                        break
                row[target] = val
            records.append(row)
 
    wide = pd.DataFrame(records)
 
    # ── Sort chronologically ──────────────────────────────────────────────────
    wide["_sort"] = wide["quarter"].apply(_sort_key)
    wide = (
        wide.sort_values(["company", "_sort"])
            .drop(columns=["_sort"])
            .reset_index(drop=True)
    )
 
    # ── Forward fill within each company ─────────────────────────────────────
    wide = (
        wide.groupby("company", group_keys=False)
            .apply(lambda x: x.ffill())
            .reset_index(drop=True)
    )
 
    # ── Derived metrics ───────────────────────────────────────────────────────
    wide["profit_margin"] = wide["net_profit"] / wide["revenue"] * 100
 
    # ── Time-series features (QoQ / YoY) ─────────────────────────────────────
    for company in wide["company"].unique():
        mask = wide["company"] == company
        idx  = wide[mask].index
        for col in _FEATURE_COLS:
            values = pd.to_numeric(wide.loc[mask, col], errors="coerce")
            wide.loc[idx, f"{col}_qoq_change"]     = _safe_pct_change(values)
            wide.loc[idx, f"{col}_qoq_abs_change"]  = values.diff()
            wide.loc[idx, f"{col}_yoy_change"]      = (
                (values - values.shift(4)) / values.shift(4) * 100
            )
 
    # ── Drop all-NaN columns ──────────────────────────────────────────────────
    wide = wide.dropna(how="all", axis=1)
 
    return wide.to_dict(orient="records")
 