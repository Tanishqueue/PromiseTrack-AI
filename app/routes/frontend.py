"""
app/routes/frontend.py
Serves the frontend index.html for all non-API routes.
"""
 
from flask import Blueprint, render_template
 
frontend_bp = Blueprint("frontend", __name__)
 
 
@frontend_bp.route("/", defaults={"path": ""})
@frontend_bp.route("/<path:path>")
def serve_frontend(path):
    return render_template("index.html")
 