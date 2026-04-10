#!/usr/bin/env python3
"""
rag_explainer.py
Retrieves relevant context from ChromaDB and generates a grounded
financial consistency explanation via the Groq LLM API.

Flask entry point : run_rag_explanation(query)
Groq client       : call load_groq_client() once in create_app()
ChromaDB          : reuses the singleton from build_vector_db.py
"""

import os
import re
from typing import Optional
from groq import Groq
from pipelines.rag.build_vector_db import get_collection


# ── Module-level singleton ────────────────────────────────────────────────────

_groq_client: Optional[Groq] = None


# ── Loader (call once in create_app()) ───────────────────────────────────────

def load_groq_client() -> None:
    global _groq_client
    if _groq_client is not None:
        return
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    _groq_client = Groq(api_key=api_key)


def is_groq_loaded() -> bool:
    return _groq_client is not None


# ── Helpers ───────────────────────────────────────────────────────────────────



def _is_noise(text: str) -> bool:
    if len(text) < 40: return True
    if "Page" in text or "Classification" in text: return True
    return False


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(query: str, company: str, top_k: int = 12) -> list[str]:
    """
    Queries ChromaDB for the most relevant transcript excerpts for a specific company.
    """
    collection   = get_collection()
    where_filter = {"company": company} if company else None

    results = collection.query(
        query_texts=[query + " financial performance growth decline guidance margins"],
        n_results=top_k * 2,
        where=where_filter,
    )

    docs = results.get("documents", [[]])[0]

    # Deduplicate and filter noise
    seen, clean = set(), []
    for d in docs:
        if d not in seen and not _is_noise(d):
            seen.add(d)
            clean.append(d)

    return clean[:top_k]


# ── LLM generation ────────────────────────────────────────────────────────────

def _generate_analysis(query: str, context: list[str]) -> dict:
    """Calls Groq LLM to synthesize the context into a structured response."""
    if not context:
        return {
            "explanation": "Insufficient transcript data available to synthesize an analysis.",
            "positive": [],
            "negative": []
        }

    context_text = "\n".join(f"- {x}" for x in context)

    prompt = f"""You are an expert financial analyst. Read the following excerpts from a company's recent earnings transcripts.

CONTEXT:
{context_text}

TASK:
1. Write a cohesive, 3-sentence executive summary of the company's performance and forward-looking outlook.
2. Identify 2 specific positive signals/tailwinds mentioned in the text.
3. Identify 2 specific negative signals/headwinds/risks mentioned in the text.

You MUST format your exact response like this:
SUMMARY: [Your 3 sentence summary here]
POS: [First positive signal]
POS: [Second positive signal]
NEG: [First negative signal]
NEG: [Second negative signal]
"""

    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    raw_text = response.choices[0].message.content.strip()

    # Parse the LLM output
    explanation = ""
    positive = []
    negative = []

    for line in raw_text.split('\n'):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            explanation = line.replace("SUMMARY:", "").strip()
        elif line.startswith("POS:"):
            positive.append(line.replace("POS:", "").strip())
        elif line.startswith("NEG:"):
            negative.append(line.replace("NEG:", "").strip())

    # Fallback if LLM didn't format perfectly
    if not explanation:
        explanation = raw_text

    return {
        "explanation": explanation,
        "positive": positive,
        "negative": negative
    }


# ── Flask entry point ─────────────────────────────────────────────────────────

def run_rag_explanation(query: str, company: str, top_k: int = 10) -> dict:
    """
    Main entry point for the Flask app.
    """
    if not is_groq_loaded():
        raise RuntimeError(
            "Groq client is not loaded. Call load_groq_client() "
            "inside create_app() before handling requests."
        )

    context = retrieve_context(query, company=company, top_k=top_k)
    analysis = _generate_analysis(query, context)

    return {
        "query":            query,
        "positive_signals": analysis["positive"],
        "negative_signals": analysis["negative"],
        "explanation":      analysis["explanation"],
    }