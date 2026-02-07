import os
from flask import Flask, render_template
from dotenv import load_dotenv

from backend.routes.analysis import analysis_bp
from backend.routes.history import history_bp

load_dotenv()


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static"
    )

    # Core config
    app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER")
    app.config["DATABASE_PATH"] = os.getenv("DATABASE_PATH")

    if not app.config["UPLOAD_FOLDER"]:
        raise RuntimeError("UPLOAD_FOLDER is not set")

    if not app.config["DATABASE_PATH"]:
        raise RuntimeError("DATABASE_PATH is not set")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # API blueprints
    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")

    # Frontend routes
    @app.route("/")
    def login():
        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/agent")
    def agent():
        return render_template("agent.html")

    # Health check
    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok", "service": "legal-doc-ai"}, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
