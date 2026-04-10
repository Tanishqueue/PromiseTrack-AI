"""
config.py
Single source of truth for all paths, model settings, and environment config.
Import from here everywhere — never hardcode paths in pipeline files.
"""
 
import os
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
 
# ── Directory layout ──────────────────────────────────────────────────────────
 
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
CHROMA_DIR   = BASE_DIR / "chroma_db"
 
# ── Model paths ───────────────────────────────────────────────────────────────
 
MODEL_PATH = str(
    BASE_DIR
    / "claim_classification_model_distilbert_trained"
    / "claim_classifier_model"
    / "checkpoint-902"
)
 
# ── External API keys ─────────────────────────────────────────────────────────
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
 
# ── Inference settings ────────────────────────────────────────────────────────
 
CLAIM_MODEL_BATCH_SIZE = 64
CLAIM_MODEL_THRESHOLD  = 0.6
CLAIM_MODEL_MAX_LENGTH = 128
 
# ── RAG settings ──────────────────────────────────────────────────────────────
 
RAG_TOP_K         = 6
CHROMA_DB_PATH    = str(CHROMA_DIR)
CHROMA_COLLECTION = "claims"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL        = "llama-3.3-70b-versatile"
 
# ── Flask settings ────────────────────────────────────────────────────────────
 
DEBUG      = os.getenv("FLASK_DEBUG", "false").lower() == "true"
HOST       = os.getenv("FLASK_HOST", "0.0.0.0")
PORT       = int(os.getenv("FLASK_PORT", 5000))
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")