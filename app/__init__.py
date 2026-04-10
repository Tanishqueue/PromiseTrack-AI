"""
app/__init__.py
Flask application factory.
"""

from flask import Flask

import config
from db import init_db
from pipelines.ml.run_claim_model  import load_claim_model
from pipelines.rag.build_vector_db import load_vector_db
from pipelines.rag.rag_explainer   import load_groq_client


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder=str(config.FRONTEND_DIR),
        static_url_path="",
    )

    app.secret_key                   = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    app.config["UPLOAD_FOLDER"]      = str(config.DATA_DIR)

    # Initialise DB schema (creates tables if they don't exist)
    init_db()

    # Load heavyweight models once at startup
    load_claim_model(config.MODEL_PATH)
    load_vector_db(config.CHROMA_DB_PATH)
    load_groq_client()

    # Register blueprints
    from app.routes.frontend import frontend_bp
    from app.routes.pipeline import pipeline_bp

    app.register_blueprint(pipeline_bp, url_prefix="/api")
    app.register_blueprint(frontend_bp)

    return app