from typing import Optional
#!/usr/bin/env python3
"""
test_retrieval.py
Diagnostic utility for inspecting ChromaDB retrieval quality.
Returns structured signal breakdown for a given query — useful for
a debug/admin route in Flask.

Flask entry point: run_retrieval_test(query, top_k)
ChromaDB         : reuses the singleton from build_vector_db.py
"""

from pipelines.rag.build_vector_db import get_collection


# ── Helpers (shared with rag_explainer.py) ────────────────────────────────────

def _detect_company(query: str) -> Optional[str]:
    q = query.lower()
    if "hdfc" in q:
        return "hdfc"
    if "reliance" in q:
        return "Reliance Industries"
    return None


def _is_noise(text: str) -> bool:
    if len(text) < 40:
        return True
    if "Page" in text or "Classification" in text:
        return True
    return False


# ── Flask entry point ─────────────────────────────────────────────────────────

def run_retrieval_test(query: str, top_k: int = 8) -> dict:
    """
    Diagnostic retrieval — returns a structured breakdown of signal quality.
    Useful for a /debug/retrieval admin route.

    Input : query  — natural language query string
            top_k  — max signals to return per category (default 8)
    Output: dict with keys:
              query            — original query
              db_count         — total documents in ChromaDB
              positive_signals — list[dict{sentence, result, reason}]
              negative_signals — list[dict{sentence, result, reason}]
              weak_signals     — list[dict{sentence, result, reason}]
    """
    collection   = get_collection()
    company      = _detect_company(query)
    where_filter = {"company": company} if company else None

    results = collection.query(
        query_texts=[query + " financial performance growth decline guidance miss"],
        n_results=top_k * 4,
        where=where_filter,
    )

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # Deduplicate and filter noise
    seen, clean = set(), []
    for d, m in zip(docs, metas):
        if d in seen or _is_noise(d):
            continue
        seen.add(d)
        clean.append((d, m))

    # Split into signal categories
    positive, negative, weak = [], [], []

    for d, m in clean:
        result = m.get("result", "")
        reason = m.get("reason", "")
        entry  = {"sentence": d, "result": result, "reason": reason}

        if result == "VERIFIED" and "STRONG" in reason:
            positive.append(entry)
        elif result == "NOT VERIFIED" and "CONTRADICTS_DIRECTION" in reason:
            negative.append(entry)
        else:
            weak.append(entry)

    return {
        "query":            query,
        "db_count":         collection.count(),
        "positive_signals": positive[:top_k],
        "negative_signals": negative[:top_k],
        "weak_signals":     weak[:top_k],
    }


# ── Batch helper (for running multiple test queries at once) ──────────────────

def run_retrieval_test_batch(queries: list[str], top_k: int = 8) -> list[dict]:
    """
    Runs run_retrieval_test() over a list of queries.
    Useful for a bulk diagnostic endpoint.
    """
    return [run_retrieval_test(q, top_k=top_k) for q in queries]