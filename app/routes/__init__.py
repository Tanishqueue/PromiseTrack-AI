"""
app/routes/__init__.py
Registers all blueprints with the Flask app.
Import and call register_blueprints(app) from create_app().
"""
 
from flask import Flask
 
 
def register_blueprints(app: Flask) -> None:
    from app.routes.frontend import frontend_bp
    from app.routes.pipeline import pipeline_bp
 
    app.register_blueprint(frontend_bp)
    app.register_blueprint(pipeline_bp, url_prefix="/api")
 