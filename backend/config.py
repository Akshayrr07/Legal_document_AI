"""
Application configuration module.
Loads and validates all environment variables required for the Legal AI system.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration class for the Legal Document AI Flask application.
    All values are sourced from environment variables with sensible defaults.
    """

    # ── Flask ──────────────────────────────────────────────────────────────────
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")

    # ── File uploads ───────────────────────────────────────────────────────────
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "data/uploads")
    ALLOWED_EXTENSIONS: set = {
        ext.strip().lower()
        for ext in os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,pdf").split(",")
    }
    # Default 16 MB; env value must be an integer (bytes)
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH_BYTES", str(16 * 1024 * 1024)))

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/results.db")

    # ── External tools ─────────────────────────────────────────────────────────
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

    # ── HuggingFace ────────────────────────────────────────────────────────────
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_HOME: str = os.path.join(os.path.dirname(__file__), "../.cache/huggingface")

    @classmethod
    def validate(cls) -> None:
        """
        Raise RuntimeError for any missing critical configuration values.
        Call this once during app creation.
        """
        required = {
            "UPLOAD_FOLDER": cls.UPLOAD_FOLDER,
            "DATABASE_PATH": cls.DATABASE_PATH,
            "TESSERACT_PATH": cls.TESSERACT_PATH,
        }
        missing = [key for key, val in required.items() if not val]
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Check your .env file."
            )
