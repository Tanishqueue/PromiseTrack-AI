# ── Stage 1: Build image ──────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps needed by some python packages (pdfplumber, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformer model at build time so it's baked in
# (avoids downloading at runtime on every cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/          app/
COPY pipelines/    pipelines/
COPY config.py     config.py
COPY db.py         db.py
COPY run.py        run.py

# ── Copy pre-built databases (generated locally before deployment) ─────────────
# These are read-only at runtime — no pipeline runs on the server.
COPY promisetrack.db  promisetrack.db
COPY chroma_db/       chroma_db/

# ── Copy the trained DistilBERT model checkpoint ──────────────────────────────
COPY claim_classification_model_distilbert_trained/ \
     claim_classification_model_distilbert_trained/

# ── Logo cache (optional — pre-warm if desired, otherwise fetched on demand) ──
COPY data/logos/   data/logos/

# ── Expose port 7860 (HuggingFace Spaces standard) ───────────────────────────
EXPOSE 7860

# ── Set env defaults (real secrets go in HF Space Settings > Secrets) ─────────
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=7860 \
    FLASK_DEBUG=false

# ── Run with Gunicorn (production-grade, matches your existing Space) ──────────
# run:app = the `app` object created in run.py
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--worker-tmp-dir", "/dev/shm", \
     "--timeout", "180", \
     "run:app"]
