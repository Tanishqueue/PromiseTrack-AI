from typing import Optional
#!/usr/bin/env python3
"""
extract_attributes.py
Extracts structured attributes (metric, direction, magnitude) from claim sentences.

Flask entry point: run_attribute_extraction_pipeline(claim_records)
"""

import re


# ── Metric map ────────────────────────────────────────────────────────────────

METRIC_MAP = [
    ("ebitda",       ["ebitda"]),
    ("revenue",      ["revenue", "sales", "turnover", "topline"]),
    ("profit",       ["net profit", "profit", "pat", "pbt", "earnings"]),
    ("margin",       ["margin", "ebitda margin", "operating margin"]),
    ("expenses",     ["cost", "expense", "capex", "opex"]),
    ("loan",         ["loan", "loan book", "credit", "advances"]),
    ("deposit",      ["deposit", "casa"]),
    ("npa",          ["npa", "gnpa", "nnpa"]),
    ("arpu",         ["arpu"]),
    ("aum",          ["aum"]),
    ("volume",       ["volume"]),
    ("subscriber",   ["subscriber", "user", "customer base"]),
    ("order",        ["order", "order book", "backlog"]),
    ("debt",         ["debt", "net debt"]),
    ("cash_flow",    ["cash flow"]),
    ("return",       ["roe", "roce", "roa"]),
    ("provision",    ["provision"]),
    ("market_share", ["market share"]),
]

# ── Direction patterns (data-driven) ─────────────────────────────────────────

INCREASE_PATTERNS = r"(increase|growth|grew|improve|expanded|rise|higher|up|strong|robust|healthy|better)"
DECREASE_PATTERNS = r"(decline|decrease|fall|drop|loss|pressure|weak|compression|down|impact)"
NEUTRAL_PATTERNS  = r"(stable|steady|flat|maintain|unchanged)"

# ── Magnitude ─────────────────────────────────────────────────────────────────

_MAG_RE = re.compile(
    r'(\d+(\.\d+)?\s*(%|bps|basis points|crore|million|billion|lakh))',
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def _split_sentence(s: str) -> list[str]:
    """Split on contrast conjunctions only — NOT on 'and'."""
    parts = re.split(r'\bbut\b|\bwhile\b|\bhowever\b', s)
    return [p.strip() for p in parts if len(p.strip()) > 20]

def extract_metrics(text: str) -> list[str]:
    t = text.lower()
    found = []
    for name, kws in METRIC_MAP:
        for kw in kws:
            if kw in t:
                found.append(name)
                break
    return list(set(found))

def extract_direction(text: str, metric: str) -> Optional[str]:
    t = text.lower()

    # Global direction — decrease takes priority
    if re.search(DECREASE_PATTERNS, t):
        return "decrease"
    if re.search(r"(increase|growth|grew|improve|expanded|rise|higher|up)", t):
        return "increase"
    # Soft positive → neutral
    if re.search(r"(strong|robust|healthy|solid)", t):
        return "neutral"
    if re.search(NEUTRAL_PATTERNS, t):
        return "neutral"

    # Metric-local fallback
    words = re.findall(r"[a-z]+", t)
    metric_positions = [i for i, w in enumerate(words) if metric in w]
    for pos in metric_positions:
        window = " ".join(words[max(0, pos - 8) : pos + 8])
        if re.search(DECREASE_PATTERNS, window):
            return "decrease"
        if re.search(INCREASE_PATTERNS, window):
            return "increase"

    return None

def extract_magnitude(text: str) -> Optional[str]:
    m = _MAG_RE.findall(text)
    return m[0][0] if m else None


# ── Flask entry point ─────────────────────────────────────────────────────────

def run_attribute_extraction_pipeline(claim_records: list[dict]) -> list[dict]:
    """
    Main entry point for the Flask app.

    Input : List of dicts from run_claim_extraction_pipeline(),
            each must have: sentence, company, quarter
    Output: List of dicts — one per (sentence-part × metric) combination.
    """
    records = []

    for row in claim_records:
        sentence = str(row.get("sentence", ""))

        for part in _split_sentence(sentence):
            clean_sent = _clean(part)
            metrics    = extract_metrics(clean_sent)

            if not metrics:
                continue

            magnitude = extract_magnitude(clean_sent)

            for metric in metrics:
                direction = extract_direction(clean_sent, metric)
                records.append({
                    "company":           row.get("company", ""),
                    "quarter":           row.get("quarter", ""),
                    "sentence":          part,
                    "metric":            metric,
                    "direction":         direction,
                    "magnitude":         magnitude,
                    "direction_missing": direction is None,
                })

    # Deduplicate on sentence + metric
    seen = set()
    unique = []
    for r in records:
        key = (r["sentence"], r["metric"])
        if key not in seen:
            unique.append(r)
            seen.add(key)

    return unique