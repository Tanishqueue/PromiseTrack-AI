#!/usr/bin/env python3
"""
build_vector_db.py
Builds (or reuses) a persistent ChromaDB vector store from verified claim records.

Flask entry points:
  load_vector_db(chroma_path)          — call once in create_app()
  build_vector_db(verified_records)    — idempotent; skips if DB already populated
  get_collection()                     — returns the live collection for querying
"""

from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer


# ── Module-level singletons ───────────────────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client:   Optional[chromadb.PersistentClient] = None
_collection       = None


# ── Loader (call once in create_app()) ───────────────────────────────────────

def load_vector_db(chroma_path: str) -> None:
    """
    Initialises the embedding model and ChromaDB client.
    Call once inside create_app() — NOT inside a request handler.

    Args:
        chroma_path: Absolute path to the ChromaDB persistence directory.
                     Use config.CHROMA_DB_PATH to supply this.
    """
    global _embedding_model, _chroma_client, _collection

    if _chroma_client is not None:
        return  # Already loaded — skip silently

    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    _chroma_client   = chromadb.PersistentClient(path=chroma_path)
    _collection      = _chroma_client.get_or_create_collection(name="claims")


def is_vector_db_loaded() -> bool:
    return _chroma_client is not None


def get_collection():
    """Returns the live ChromaDB collection for use in rag_explainer.py."""
    if _collection is None:
        raise RuntimeError(
            "Vector DB is not loaded. Call load_vector_db(config.CHROMA_DB_PATH) "
            "inside create_app() before handling requests."
        )
    return _collection


# ── Flask entry point ─────────────────────────────────────────────────────────

def build_vector_db(verified_records: list, company: str = None) -> dict:
    """
    Adds claim records to ChromaDB. Appends per company — does not skip
    if DB already has data from other companies.

    Input : verified_records — list[dict] with sentence, company, quarter, result, reason
            company          — company name for dedup check (optional)
    Output: dict with keys: status, count
    """
    if not is_vector_db_loaded():
        raise RuntimeError(
            "Vector DB is not loaded. Call load_vector_db(config.CHROMA_DB_PATH) "
            "inside create_app() before handling requests."
        )

    if not verified_records:
        return {"status": "empty_input", "count": 0}

    # Check if THIS company's data already exists (not the whole collection)
    if company:
        try:
            existing = _collection.get(where={"company": company})
            if existing and len(existing.get("ids", [])) > 0:
                return {"status": "skipped", "count": _collection.count()}
        except Exception:
            pass

    # Prepare documents
    documents, metadatas, ids = [], [], []

    for i, row in enumerate(verified_records):
        text = str(row.get("sentence", "")).strip()
        if not text:
            continue

        documents.append(text)
        metadatas.append({
            "company": str(row.get("company", "")),
            "quarter": str(row.get("quarter", "")),
            "result":  str(row.get("result",  "")),
            "reason":  str(row.get("reason",  "")),
        })
        comp_key = str(row.get("company", "unknown")).replace(" ", "_")[:30]
        ids.append(f"{comp_key}_{i}")

    if not documents:
        return {"status": "empty_input", "count": 0}

    embeddings = _embedding_model.encode(documents).tolist()

    _collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return {"status": "built", "count": _collection.count()}