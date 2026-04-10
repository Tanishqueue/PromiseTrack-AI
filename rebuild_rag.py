#!/usr/bin/env python3
"""
rebuild_rag.py
Rebuilds ChromaDB from claims stored in SQLite and regenerates
the RAG explanation + analysis cache for all companies.
Run this instead of the full pipeline when only ChromaDB/RAG needs fixing.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from db import init_db, get_db
from app.services.cache_service import get_ready_companies, get_claims, save_analysis
from pipelines.rag.build_vector_db import load_vector_db, build_vector_db
from pipelines.rag.rag_explainer import load_groq_client, run_rag_explanation

def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def _signal_title(sentence):
    words = sentence.split()
    return " ".join(words[:8]).rstrip(",.;") + ("…" if len(words) > 8 else "")

def _infer_verdict(pos, neg):
    if len(pos) > len(neg) * 1.5: return "Positive"
    if len(neg) > len(pos) * 1.5: return "Negative"
    return "Mixed"

def main():
    _log("Loading models...")
    init_db()
    load_vector_db(config.CHROMA_DB_PATH)
    load_groq_client()
    _log("Models ready.\n")

    companies = get_ready_companies()
    _log(f"Found {len(companies)} ready companies.\n")

    for company in companies:
        cid   = company["id"]
        name  = company["display_name"]
        folder = company["folder_name"]
        _log(f"Processing: {name}")

        # 1. Load claims from SQLite
        claims = get_claims(cid)
        if not claims:
            _log(f"  ⚠ No claims found in DB — skipping.\n")
            continue

        # 2. Build ChromaDB entries from claims + add company field
        docs_for_vector = [
            {
                "sentence": c.get("sentence", ""),
                "company":  name,
                "quarter":  c.get("quarter", ""),
                "result":   c.get("result", "UNVERIFIED"),
                "reason":   "CLAIM",
            }
            for c in claims if c.get("sentence")
        ]

        _log(f"  → Loading {len(docs_for_vector)} claims into ChromaDB...")
        status = build_vector_db(docs_for_vector, company=name)
        _log(f"  → Vector DB: {status}")

        # 3. Re-run RAG explanation
        _log(f"  → Generating RAG explanations...")
        query = f"{name} financial performance earnings guidance"

        # Load current cached analysis to get existing metrics/claims
        with get_db() as conn:
            row = conn.execute(
                "SELECT result_json FROM analysis_cache WHERE company_id=? AND mode='full'",
                (cid,)
            ).fetchone()

        import json
        existing = json.loads(row["result_json"]) if row else {}

        for mode in ("full", "earnings", "financial"):
            try:
                rag     = run_rag_explanation(query, company=name, top_k=config.RAG_TOP_K)
                pos_raw = rag.get("positive_signals", [])
                neg_raw = rag.get("negative_signals", [])

                result = {
                    **existing,               # keep metrics, claims, timeseries etc.
                    "mode":             mode,
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
                }
                save_analysis(cid, mode, result)
            except Exception as e:
                _log(f"  ⚠ Mode '{mode}' failed: {e}")

        _log(f"  ✓ {name} done.\n")

    _log("All done.")

if __name__ == "__main__":
    main()
