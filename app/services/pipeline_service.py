"""
app/services/pipeline_service.py
Serves analysis results from SQLite cache.
No pipeline logic runs here — all processing happens in process_company.py.
"""

import difflib
from app.services.cache_service import (
    get_ready_companies,
    get_cached_analysis,
)


def get_known_companies() -> list:
    """
    Returns list of processed companies for the autocomplete dropdown.
    Only companies with status='ready' are returned.
    Automatically reflects new companies added via process_company.py.
    """
    return [
        {"display": c["display_name"], "folder": c["folder_name"]}
        for c in get_ready_companies()
    ]


def _fuzzy_match_company(query: str):
    """
    Finds the best matching company from the DB for a user query.
    Tries exact → substring → difflib fuzzy in order.
    Returns the full company DB row or None.
    """
    companies = get_ready_companies()
    if not companies:
        return None

    query_lower = query.strip().lower()

    lookup = {}
    for c in companies:
        lookup[c["display_name"].lower()] = c
        lookup[c["folder_name"].lower()]  = c

    if query_lower in lookup:
        return lookup[query_lower]

    for key, company in lookup.items():
        if query_lower in key or key.startswith(query_lower):
            return company

    matches = difflib.get_close_matches(query_lower, lookup.keys(), n=1, cutoff=0.5)
    if matches:
        return lookup[matches[0]]

    return None


def analyse_company(company_name: str, mode: str) -> dict:
    """
    Looks up cached analysis for the best matching company.
    Returns the cached result dict, or an informative error dict if not found.
    """
    company = _fuzzy_match_company(company_name)

    if not company:
        return {
            "error":   "not_found",
            "message": (
                f"No processed data found for '{company_name}'. "
                f"Run: python3 process_company.py \"{company_name}\""
            ),
            "company": company_name,
            "mode":    mode,
        }

    result = get_cached_analysis(company["id"], mode)

    if not result:
        return {
            "error":   "not_cached",
            "message": (
                f"'{company['display_name']}' is registered but mode '{mode}' "
                f"has not been cached yet. "
                f"Run: python3 process_company.py \"{company['display_name']}\""
            ),
            "company": company["display_name"],
            "mode":    mode,
        }

    return result

def chat_with_company(company_name: str, query: str) -> dict:
    """
    Handles interactive RAG chat by querying ChromaDB directly and passing context to Groq LLM.
    """
    company = _fuzzy_match_company(company_name)

    if not company:
        return {
            "error":   "not_found",
            "message": f"No processed data found for '{company_name}'.",
            "company": company_name,
        }

    display_name = company["display_name"]

    try:
        import config
        import chromadb
        from groq import Groq

        # 1. Query ChromaDB for relevant claims
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION)

        results = collection.query(
            query_texts=[query],
            n_results=config.RAG_TOP_K,
            where={"company": display_name}
        )

        # If no context is found
        if not results or not results.get("documents") or not results["documents"][0]:
            return {"answer": f"I couldn't find any specific claims or financial data regarding your query for {display_name} in my database."}

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # 2. Format context for the LLM
        context_blocks = []
        for doc, meta in zip(documents, metadatas):
            q = meta.get("quarter", "Unknown Qtr")
            res = meta.get("result", "UNVERIFIED")
            context_blocks.append(f"[{q}] Claim ({res}): {doc}")

        context_str = "\n".join(context_blocks)

        # 3. Ask Groq LLM
        client = Groq(api_key=config.GROQ_API_KEY)
        prompt = f"""You are a financial AI assistant answering questions about {display_name}.
Use ONLY the following extracted claims and verified outcomes to answer the question.
If the context doesn't contain the answer, explicitly state that you don't have enough information.
Keep your answer concise, analytical, and professional.

CONTEXT:
{context_str}

QUESTION:
{query}
"""

        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )

        return {"answer": completion.choices[0].message.content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "rag_error", "answer": f"Backend connected, but RAG encountered an error: {str(e)}"}