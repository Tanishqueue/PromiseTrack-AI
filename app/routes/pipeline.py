"""
app/routes/pipeline.py
All API endpoints. Every route delegates to pipeline_service — no pipeline
logic lives here.
"""

from flask import Blueprint, jsonify, request

from app.services.pipeline_service import analyse_company, get_known_companies
from pipelines.ml.run_claim_model  import is_model_loaded
from pipelines.rag.build_vector_db import is_vector_db_loaded, build_vector_db

pipeline_bp = Blueprint("pipeline", __name__)


# ── Health check ──────────────────────────────────────────────────────────────

@pipeline_bp.get("/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     is_model_loaded(),
        "vector_db_loaded": is_vector_db_loaded(),
    })


# ── Companies list (for autocomplete) ────────────────────────────────────────

@pipeline_bp.get("/companies")
def companies():
    return jsonify(get_known_companies())


# ── Main analysis endpoint ────────────────────────────────────────────────────

@pipeline_bp.post("/analyse")
def analyse():
    body    = request.get_json(silent=True) or {}
    company = (body.get("company") or "").strip()
    mode    = (body.get("mode")    or "full").strip().lower()

    if not company:
        return jsonify({"error": "company is required"}), 400
    if mode not in ("full", "earnings", "financial"):
        return jsonify({"error": "mode must be full | earnings | financial"}), 400

    try:
        result = analyse_company(company, mode)
        # Cache miss — return 404 with helpful message
        if "error" in result and result["error"] in ("not_found", "not_cached"):
            return jsonify(result), 404
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Vector DB query endpoint ─────────────────────────────────────────────────

@pipeline_bp.post("/chat")
def chat():
    body    = request.get_json(silent=True) or {}
    company = (body.get("company") or "").strip()
    query   = (body.get("query") or "").strip()

    if not company:
        return jsonify({"error": "company is required"}), 400
    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        from app.services.pipeline_service import chat_with_company
        result = chat_with_company(company, query)
        if "error" in result and result["error"] == "not_found":
            return jsonify(result), 404
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ── Vector DB build endpoint (admin / one-time setup) ─────────────────────────

@pipeline_bp.post("/admin/build-vector-db")
def admin_build_vector_db():
    body    = request.get_json(silent=True) or {}
    records = body.get("verified_records", [])
    if not records:
        return jsonify({"error": "verified_records is required"}), 400
    try:
        status = build_vector_db(records)
        return jsonify(status)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Logo cache endpoint ──────────────────────────────────────────────────────

@pipeline_bp.get("/logo/<company_name>")
def get_company_logo(company_name):
    import os
    import requests
    from flask import send_file
    
    LOGO_DEV_PUBLIC_KEY = 'pk_MmKxh9tYQ1Gvapx5RFmcTA'
    LOGO_CACHE_DIR = os.path.join("data", "logos")
    os.makedirs(LOGO_CACHE_DIR, exist_ok=True)
    
    # We use a safe filename
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', company_name)
    cache_path = os.path.join(LOGO_CACHE_DIR, f"{safe_name}.png")
    
    if os.path.exists(cache_path):
        return send_file(os.path.abspath(cache_path), mimetype='image/png')
        
    DOMAIN_MAP = {
        # Current folder names (updated after data reorganization)
        "INFOSYS":              "infosys.com",
        "BAJAJ FINANCE":        "bajajfinserv.in",
        "Kotak Mahindra Bank":  "kotakbank.com",
        "SUN PHARMA":           "sunpharma.com",
        "Reliance  Industries": "ril.com",
        "Reliance Industries":  "ril.com",
        "TCS":                  "tcs.com",
        "HDFC Bank":            "hdfcbank.com",
        "Mahindra & Mahindra":  "mahindra.com",
        "SBI":                  "sbi.co.in",
        "ITC":                  "itcportal.com",
        "L&T":                  "larsentoubro.com",
        "ICICI Bank":           "icicibank.com",
        "HCL":                  "hcltech.com",
        "Bharthi Airtel":       "airtel.in",
        "AXIS Bank":            "axisbank.com",
    }

    try:
        if company_name in DOMAIN_MAP:
            domain = DOMAIN_MAP[company_name]
            url = f"https://img.logo.dev/{domain}?size=120&token={LOGO_DEV_PUBLIC_KEY}"
        else:
            import urllib.parse
            encoded_name = urllib.parse.quote(company_name)
            url = f"https://img.logo.dev/name/{encoded_name}?size=120&token={LOGO_DEV_PUBLIC_KEY}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            return send_file(os.path.abspath(cache_path), mimetype='image/png')
        else:
            return jsonify({"error": "Logo not found"}), response.status_code
            
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500