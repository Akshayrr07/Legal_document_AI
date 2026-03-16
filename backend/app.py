"""
Flask application factory for the Legal Document AI system.
Configures structured logging, blueprints, file size limits and frontend routes.
"""

import os
import logging
from flask import Flask, render_template

from backend.config import Config


def _configure_logging(app: Flask) -> None:
    """Set up structured logging for both Flask and the root logger."""
    log_level = logging.DEBUG if app.config["FLASK_ENV"] == "development" else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.setLevel(log_level)
    app.logger.info("Logging configured at level: %s", logging.getLevelName(log_level))


def create_app() -> Flask:
    """
    Application factory.
    Creates, configures and returns the Flask app instance.
    """
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )

    # ── Load & validate configuration ──────────────────────────────────────────
    Config.validate()
    app.config.from_object(Config)

    _configure_logging(app)

    # Ensure the uploads directory exists at startup
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.logger.info("Upload folder ready: %s", app.config["UPLOAD_FOLDER"])

    # ── Register API blueprints ────────────────────────────────────────────────
    # Import here to avoid circular import issues during testing
    from backend.routes.analysis import analysis_bp
    from backend.routes.history import history_bp

    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.logger.info("API blueprints registered.")

    # ── Frontend routes ────────────────────────────────────────────────────────
    @app.route("/")
    def login():
        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/agent")
    def agent():
        return render_template("agent.html")

    # ── Health check ───────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok", "service": "legal-doc-ai"}, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=(Config.FLASK_ENV == "development"))
