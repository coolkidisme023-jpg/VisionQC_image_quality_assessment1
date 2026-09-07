"""
VisionQC Flask application factory.

Environment note: the original spec requested FastAPI. This development
sandbox has no outbound network access (see README / SUBMISSION_NOTES for
details), so FastAPI could not be installed or verified here. Flask
(already available in the environment) is used instead, structured with
the same separation of concerns FastAPI would encourage: a thin route
layer, a service layer (``app/services/analyzer.py``) containing all
business logic, and a persistence layer (``app/database.py``). Every
route returns the same structured JSON envelope described in the README,
so a future FastAPI port would only need to change the route layer.
"""

from __future__ import annotations

import logging

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import settings
from app.database import get_db
from app.routes.analyses import bp as analyses_bp
from app.routes.docs import bp as docs_bp
from app.routes.health import bp as health_bp


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH
    app.url_map.strict_slashes = False

    # Initialize the database (creates the SQLite file + schema if missing).
    get_db()

    app.register_blueprint(health_bp)
    app.register_blueprint(analyses_bp)
    app.register_blueprint(docs_bp)

    _register_cors(app)
    _register_error_handlers(app)

    @app.route("/")
    def index():
        return jsonify({
            "service": "VisionQC API",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
        })

    return app


def _register_cors(app: Flask) -> None:
    """Minimal, dependency-free CORS support (flask-cors is not available
    in this offline sandbox). Restricted to the configured origins."""

    allowed_origins = set(settings.FRONTEND_ORIGINS)

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        origin = request.headers.get("Origin")
        if origin and (origin in allowed_origins or "*" in allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.route("/api/<path:_any>", methods=["OPTIONS"])
    def cors_preflight(_any):  # noqa: ANN001
        return "", 204


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc):
        max_mb = settings.MAX_CONTENT_LENGTH // (1024 * 1024)
        return jsonify({
            "error": f"Uploaded file exceeds the maximum allowed size of {max_mb} MB.",
            "status": 413,
        }), 413

    @app.errorhandler(404)
    def handle_not_found(_exc):
        return jsonify({"error": "Resource not found.", "status": 404}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(_exc):
        return jsonify({"error": "Method not allowed.", "status": 405}), 405

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        return jsonify({"error": exc.description, "status": exc.code}), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        app.logger.exception("Unhandled exception")
        return jsonify({"error": "Internal server error.", "status": 500}), 500


# WSGI entry point (used by `flask run`, gunicorn/waitress, and Docker).
app = create_app()


if __name__ == "__main__":
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)
