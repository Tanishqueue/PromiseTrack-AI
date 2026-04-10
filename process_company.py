#!/usr/bin/env python3
"""
process_company.py
One-time processing script per company.
Handles PDF-only companies and PDF+XBRL companies automatically.

Usage:
    python3 process_company.py "AXIS"
    python3 process_company.py --all
    python3 process_company.py --list
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from db import init_db
from app.services.cache_service import (
    upsert_company, save_claims, save_timeseries,
    save_risk, save_analysis, get_all_companies,
)
from pipelines.text.extract_text_data         import run_text_extraction_pipeline
from pipelines.text.split_sentences            import run_sentence_splitting_pipeline
from pipelines.text.claim_extractor            import run_claim_extraction_pipeline
from pipelines.ml.run_claim_model              import load_claim_model, run_claim_model_pipeline
from pipelines.ml.extract_attributes           import run_attribute_extraction_pipeline
from pipelines.finance.extract_xbrl_data       import run_xbrl_extraction_pipeline
from pipelines.finance.prepare_timeseries_data import run_timeseries_pipeline
from pipelines.finance.verify_claims           import run_claim_verification_pipeline
from pipelines.risk.aggregate_risk             import run_risk_aggregation_pipeline
from pipelines.rag.build_vector_db             import load_vector_db, build_vector_db
from pipelines.rag.rag_explainer               import load_groq_client, run_rag_explanation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _clean_display(folder_name: str) -> str:
    name = re.sub(r'\s*\d{4}\s*Quarterly\s*Data\s*', '', folder_name, flags=re.IGNORECASE)
    return name.strip()


def _find_folder(query: str):
    import difflib
    companies_dir = Path(config.DATA_DIR) / "raw" / "companies"
    if not companies_dir.exists():
        return None
    folders = [f for f in companies_dir.iterdir() if f.is_dir()]
    query_lower = query.strip().lower()
    for f in folders:
        if f.name.lower() == query_lower or _clean_display(f.name).lower() == query_lower:
            return f
    for f in folders:
        if query_lower in f.name.lower() or query_lower in _clean_display(f.name).lower():
            return f
    names = [f.name.lower() for f in folders]
    matches = difflib.get_close_matches(query_lower, names, n=1, cutoff=0.5)
    if matches:
        return folders[names.index(matches[0])]
    return None


def _has_xml(folder: Path) -> bool:
    """Check if company folder has any XML files (any depth)."""
    return any(folder.rglob("*.xml"))


def _match_records(records: list, folder_name: str) -> list:
    folder_lower  = folder_name.lower()
    display_lower = _clean_display(folder_name).lower()
    exact = [r for r in records
             if r.get("company", "").lower() in (folder_lower, display_lower)]
    if exact:
        return exact
    return [r for r in records
            if display_lower in r.get("company", "").lower()
            or r.get("company", "").lower() in folder_lower]


def _signal_title(sentence: str) -> str:
    words = sentence.split()
    return " ".join(words[:8]).rstrip(",.;") + ("…" if len(words) > 8 else "")


def _infer_verdict(pos: list, neg: list) -> str:
    if len(pos) > len(neg) * 1.5:
        return "Positive"
    if len(neg) > len(pos) * 1.5:
        return "Negative"
    return "Mixed"


def _fmt(value, suffix="") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.1f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _clean_quarter(q: str) -> str:
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


def _trend(change) -> str:
    if change is None:
        return "flat"
    try:
        v = float(change)
        return "up" if v > 1 else ("down" if v < -1 else "flat")
    except (TypeError, ValueError):
        return "flat"


def _format_metrics(ts_records: list) -> list:
    if not ts_records:
        return []
    row = sorted(ts_records, key=lambda r: str(r.get("quarter", "")))[-1]
    defs = [
        ("Revenue",          "revenue",          "₹ Cr", "revenue_qoq_change"),
        ("Net Profit",       "net_profit",        "₹ Cr", "net_profit_qoq_change"),
        ("Operating Profit", "operating_profit",  "₹ Cr", "operating_profit_qoq_change"),
        ("Profit Margin",    "profit_margin",     "%",    "profit_margin_qoq_change"),
    ]
    out = []
    for label, key, unit, chg_key in defs:
        val = row.get(key)
        if val is None:
            continue
        out.append({
            "label": f"{label} · {row.get('quarter', '')}",
            "value": _fmt(val, f" {unit}"),
            "sub":   f"QoQ {_fmt(row.get(chg_key), '%')}",
            "trend": _trend(row.get(chg_key)),
        })
    return out


def _format_bar_metrics(ts_records: list) -> list:
    if not ts_records:
        return []
    row = sorted(ts_records, key=lambda r: str(r.get("quarter", "")))[-1]
    color_map = {"up": "var(--accent-green)", "down": "var(--accent-red)", "flat": "var(--accent-blue)"}
    defs = [
        ("Revenue QoQ %",          "revenue_qoq_change"),
        ("Net Profit QoQ %",       "net_profit_qoq_change"),
        ("Operating Profit QoQ %", "operating_profit_qoq_change"),
        ("Profit Margin QoQ %",    "profit_margin_qoq_change"),
    ]
    out = []
    for label, key in defs:
        val = row.get(key)
        if val is None:
            continue
        try:
            pct = float(val)
        except (TypeError, ValueError):
            continue
        t = _trend(pct)
        out.append({
            "label":      label,
            "target_pct": min(int(abs(pct) * 2), 100),
            "color":      color_map[t],
            "value":      _fmt(pct, "%"),
        })
    return out


def _format_risk_from_claims(attributes: list) -> dict:
    """
    Lightweight risk estimate from claim attributes alone (no XBRL needed).
    Uses direction distribution as a proxy for consistency.
    """
    if not attributes:
        return {"level": "UNKNOWN", "consistency_score": None, "warning_flag": 0}

    directions = [a.get("direction") for a in attributes if a.get("direction")]
    if not directions:
        return {"level": "UNKNOWN", "consistency_score": None, "warning_flag": 0}

    increase_rate = directions.count("increase") / len(directions)
    decrease_rate = directions.count("decrease") / len(directions)
    missing_rate  = sum(1 for a in attributes if a.get("direction_missing")) / len(attributes)

    # Simple heuristic score
    score = round(max(min(increase_rate - decrease_rate * 0.5 - missing_rate * 0.2, 1.0), 0.0), 2)
    level = "LOW" if score > 0.6 else ("MODERATE" if score > 0.3 else "HIGH")

    return {
        "level":             level,
        "consistency_score": score,
        "warning_flag":      1 if score < 0.3 else 0,
    }


def _format_risk_from_verified(risk_records: list) -> dict:
    if not risk_records:
        return {"level": "UNKNOWN", "consistency_score": None, "warning_flag": 0}
    latest = sorted(risk_records, key=lambda r: str(r.get("quarter", "")))[-1]
    score  = latest.get("consistency_score", 0)
    warn   = latest.get("warning_flag", 0)
    level  = "HIGH" if (warn or score < 0.4) else ("MODERATE" if score < 0.65 else "LOW")
    return {
        "level":             level,
        "consistency_score": score,
        "warning_flag":      warn,
        "verification_rate": latest.get("verification_rate"),
        "failure_rate":      latest.get("failure_rate"),
    }


# ── Core processing ───────────────────────────────────────────────────────────

def process_company(folder: Path) -> bool:
    folder_name  = folder.name
    display_name = _clean_display(folder_name)

    _log(f"Processing: {display_name}")
    company_id = upsert_company(folder_name, status="processing")

    try:
        companies_parent = str(folder.parent)
        attributes   = []
        ts_records   = []
        verified     = []
        risk_records = []
        sentences    = []

        # ── 1. Text pipeline (always runs) ────────────────────────────────
        _log("  → Extracting text from transcripts...")
        raw_records  = run_text_extraction_pipeline(companies_parent)
        company_docs = _match_records(raw_records, folder_name)

        if company_docs:
            _log(f"  → {len(company_docs)} file(s) found. Splitting sentences...")
            sentences    = run_sentence_splitting_pipeline(company_docs)
            _log(f"  → {len(sentences)} sentences. Running claim extraction...")
            spacy_claims = run_claim_extraction_pipeline(sentences)
            bert_claims  = run_claim_model_pipeline(sentences)
            claims       = bert_claims if bert_claims else spacy_claims
            _log(f"  → {len(claims)} claims found. Extracting attributes...")
            attributes   = run_attribute_extraction_pipeline(claims)
            _log(f"  → {len(attributes)} attribute records extracted.")
        else:
            _log("  ⚠ No transcript files found.")

        # ── 2. Financial data via XBRL/XML ────────────────────────────────
        _log("  → Extracting quarterly financials from local XBRL/XML files...")
        has_xbrl = False
        if _has_xml(folder):
            try:
                # Extract XBRL records (filtering for this specific company)
                all_fin_records = run_xbrl_extraction_pipeline(companies_parent)
                fin_records = [r for r in all_fin_records if r.get("company") == folder_name]
                
                if fin_records:
                    ts_records = run_timeseries_pipeline(fin_records)
                    save_timeseries(company_id, ts_records)
                    _log(f"  → {len(ts_records)} timeseries rows saved.")
                    has_xbrl = True

                    if attributes:
                        _log("  → Verifying claims against financials...")
                        verified     = run_claim_verification_pipeline(attributes, ts_records)
                        risk_records = run_risk_aggregation_pipeline(verified)
                        save_risk(company_id, risk_records)
                        _log(f"  → {len(verified)} claims verified.")
                else:
                    _log("  ⚠ No financial data parsed from XMLs — skipping verification.")
            except Exception as e:
                _log(f"  ⚠ Financials error: {e}")
        else:
            _log("  ⚠ No XML files found in folder — skipping verification.")

        # ── 3. Save claims to DB ──────────────────────────────────────────
        # Use verified claims if available, otherwise raw attributes
        claims_to_save = verified if verified else attributes
        if claims_to_save:
            save_claims(company_id, claims_to_save)

        # ── 4. Load ALL chunks into ChromaDB ─────────────────────────────
        _log("  → Loading ALL transcript chunks into vector DB...")
        if sentences:
            # We use 'sentences' instead of 'verified' so RAG has full context
            for s in sentences:
                s.setdefault("result", "CONTEXT")
                s.setdefault("reason", "TRANSCRIPT_EXCERPT")
            status = build_vector_db(sentences, company=display_name)
            _log(f"  → Vector DB: {status}")
        else:
            _log("  ⚠ No text chunks to load into vector DB.")

        # ── 5. RAG + cache all 3 modes ────────────────────────────────────
        _log("  → Generating RAG explanations...")
        query = f"{display_name} financial performance earnings"

        # Risk dict
        risk_dict = (
            _format_risk_from_verified(risk_records)
            if risk_records
            else _format_risk_from_claims(attributes)
        )

        # Display claims for UI (top 20) with clean quarter formats
        display_claims = [
            {
                "quarter":    _clean_quarter(a.get("quarter", "")),
                "sentence":   str(a.get("sentence", ""))[:140],
                "metric":     a.get("metric", ""),
                "direction":  a.get("direction") or "neutral",
                "result":     a.get("result", "UNVERIFIED"),
                "confidence": a.get("actual_change") or 0.0,
            }
            for a in (verified or attributes)[:20]
        ]

        quarters = sorted({
            _clean_quarter(r.get("quarter", "")) for r in (verified or attributes)
            if r.get("quarter")
        })

        for mode in ("full", "earnings", "financial"):
            try:
                rag     = run_rag_explanation(query, company=display_name, top_k=config.RAG_TOP_K)
                pos_raw = rag.get("positive_signals", [])
                neg_raw = rag.get("negative_signals", [])

                result = {
                    "company":          display_name,
                    "mode":             mode,
                    "matched_folder":   folder_name,
                    "has_xbrl":         has_xbrl,
                    "positive_signals": [
                        {"index": f"P{i+1}", "title": _signal_title(s), "body": s}
                        for i, s in enumerate(pos_raw)
                    ],
                    "negative_signals": [
                        {"index": f"N{i+1}", "title": _signal_title(s), "body": s}
                        for i, s in enumerate(neg_raw)
                    ],
                    "explanation": rag.get("explanation", ""),
                    "verdict":     _infer_verdict(pos_raw, neg_raw),
                    "metrics":     _format_metrics(ts_records),
                    "bar_metrics": _format_bar_metrics(ts_records),
                    "risk":        risk_dict,
                    "claims":      display_claims,
                    "source_label": (
                        f"{display_name} · {', '.join(quarters)}"
                        if quarters else display_name
                    ),
                }
                save_analysis(company_id, mode, result)
            except Exception as exc:
                _log(f"  ⚠ Mode '{mode}' failed: {exc}")

        # ── 6. Mark ready ─────────────────────────────────────────────────
        upsert_company(folder_name, status="ready")
        _log(f"  ✓ {display_name} done.\n")
        return True

    except Exception as exc:
        upsert_company(folder_name, status="error", error_msg=str(exc))
        _log(f"  ✗ Failed: {exc}\n")
        import traceback; traceback.print_exc()
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_list():
    companies = get_all_companies()
    if not companies:
        print("No companies registered yet.")
        return
    print(f"\n{'Display Name':<30} {'Status':<12} {'Processed At'}")
    print("-" * 65)
    for c in companies:
        print(f"{c['display_name']:<30} {c['status']:<12} {c['processed_at'] or '—'}")
    print()


def cmd_process_all():
    companies_dir = Path(config.DATA_DIR) / "raw" / "companies"
    if not companies_dir.exists():
        print(f"Not found: {companies_dir}")
        return
    folders = [f for f in sorted(companies_dir.iterdir()) if f.is_dir()]
    _log(f"Found {len(folders)} company folders.")
    ok = fail = 0
    for folder in folders:
        if process_company(folder):
            ok += 1
        else:
            fail += 1
    _log(f"Done. {ok} succeeded, {fail} failed.")


def main():
    _log("Loading models...")
    load_claim_model(config.MODEL_PATH)
    load_vector_db(config.CHROMA_DB_PATH)
    load_groq_client()
    init_db()
    _log("Ready.\n")

    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("company", nargs="?", help="Company name")
    group.add_argument("--all",  action="store_true")
    group.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        cmd_list(); return
    if args.all:
        cmd_process_all(); return

    folder = _find_folder(args.company)
    if not folder:
        print(f"No matching folder for: '{args.company}'")
        companies_dir = Path(config.DATA_DIR) / "raw" / "companies"
        if companies_dir.exists():
            print("Available:")
            for f in sorted(companies_dir.iterdir()):
                if f.is_dir():
                    print(f"  {f.name}")
        sys.exit(1)

    process_company(folder)


if __name__ == "__main__":
    main()