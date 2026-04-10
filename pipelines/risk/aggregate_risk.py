#!/usr/bin/env python3
"""
aggregate_risk.py
Aggregates claim-level verification into company-quarter risk scores.
Applies recency weights — recent quarters count more toward final score.

Flask entry point: run_risk_aggregation_pipeline(verified_records)
"""

import numpy as np
import pandas as pd


def quarter_sort_key(q: str):
    try:
        year, qn = str(q).split("-Q")
        return (int(year), int(qn))
    except Exception:
        return (0, 0)


def _assign_quarter_weights(quarters: list) -> dict:
    """Most recent quarter → 1.0, each older → -0.25, floor 0.1."""
    sorted_q = sorted(set(quarters), key=quarter_sort_key, reverse=True)
    return {q: round(max(1.0 - i * 0.25, 0.1), 2)
            for i, q in enumerate(sorted_q)}


def aggregate_group(grp: pd.DataFrame, quarter_weight: float = 1.0) -> dict:
    total = len(grp)
    if total == 0:
        return {}

    result_col = grp["result"].fillna("SKIPPED").str.upper()
    reason_col = (grp["reason"].fillna("").str.upper()
                  if "reason" in grp.columns
                  else pd.Series([""] * total))

    verified     = (result_col == "VERIFIED").sum()
    partial      = (result_col == "PARTIAL").sum()
    not_verified = (result_col == "NOT VERIFIED").sum()
    skipped      = (result_col == "SKIPPED").sum()

    verification_rate      = verified     / total
    failure_rate           = not_verified / total
    partial_rate           = partial      / total
    skipped_rate           = skipped      / total
    direction_mismatch_rate = reason_col.str.contains("CONTRADICTS_DIRECTION", na=False).sum() / total
    missing_data_rate       = reason_col.str.contains("MISSING", na=False).sum() / total
    unknown_direction_rate  = reason_col.str.contains("UNKNOWN_DIRECTION", na=False).sum() / total

    # Base consistency score weighted by quarter recency
    base_score = (
        0.5 * verification_rate
        + 0.2 * (1 - failure_rate)
        + 0.2 * (1 - direction_mismatch_rate)
        + 0.1 * (1 - missing_data_rate)
    )
    consistency_score = float(np.clip(base_score * quarter_weight, 0.0, 1.0))

    return {
        "total_claims":             total,
        "verified_count":           int(verified),
        "partial_count":            int(partial),
        "not_verified_count":       int(not_verified),
        "skipped_count":            int(skipped),
        "verification_rate":        round(verification_rate,        4),
        "failure_rate":             round(failure_rate,             4),
        "partial_rate":             round(partial_rate,             4),
        "skipped_rate":             round(skipped_rate,             4),
        "direction_mismatch_rate":  round(direction_mismatch_rate,  4),
        "missing_data_rate":        round(missing_data_rate,        4),
        "unknown_direction_rate":   round(unknown_direction_rate,   4),
        "consistency_score":        round(consistency_score,        4),
        "quarter_weight":           round(quarter_weight,           2),
    }


def run_risk_aggregation_pipeline(verified_records: list) -> list:
    """
    Input : list[dict] from run_claim_verification_pipeline()
    Output: list[dict] — one per (company, quarter), weighted by recency.
    """
    if not verified_records:
        return []

    df = pd.DataFrame(verified_records)
    df.columns = df.columns.str.strip().str.lower()
    if "result" not in df.columns:
        raise ValueError("verified_records must contain a 'result' field.")
    if "reason" not in df.columns:
        df["reason"] = ""

    # Assign recency weights across all quarters in dataset
    all_quarters   = df["quarter"].dropna().unique().tolist()
    quarter_weights = _assign_quarter_weights(all_quarters)

    records = []
    for (company, quarter), grp in df.groupby(["company", "quarter"], sort=False):
        weight = quarter_weights.get(quarter, 0.1)
        agg    = aggregate_group(grp, quarter_weight=weight)
        if not agg:
            continue
        agg["company"] = company
        agg["quarter"] = quarter
        records.append(agg)

    if not records:
        return []

    out = pd.DataFrame(records)
    out["_sort_key"] = out["quarter"].apply(quarter_sort_key)
    out = out.sort_values(["company", "_sort_key"]).reset_index(drop=True)
    out = out.drop(columns=["_sort_key"])

    # Risk drift (rolling 3-quarter average of failure_rate)
    out["risk_drift"] = 0.0
    for company, grp_idx in out.groupby("company").groups.items():
        grp_rows    = out.loc[grp_idx, "failure_rate"].reset_index(drop=True)
        rolling_avg = grp_rows.rolling(window=3, min_periods=1).mean().shift(1)
        drift       = grp_rows - rolling_avg.fillna(grp_rows)
        drift.iloc[0] = 0.0
        out.loc[grp_idx, "risk_drift"] = drift.values

    out["risk_drift"]   = out["risk_drift"].fillna(0.0).round(4)
    out["warning_flag"] = (
        (out["failure_rate"]            > 0.4) |
        (out["direction_mismatch_rate"] > 0.3) |
        (out["consistency_score"]       < 0.5)
    ).astype(int)

    final_cols = [
        "company", "quarter", "total_claims", "quarter_weight",
        "verification_rate", "failure_rate", "partial_rate", "skipped_rate",
        "direction_mismatch_rate", "missing_data_rate", "unknown_direction_rate",
        "consistency_score", "risk_drift", "warning_flag",
    ]
    out = out[[c for c in final_cols if c in out.columns]]
    return out.to_dict(orient="records")