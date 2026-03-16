"""
Analysis API route — orchestrates the full legal document processing pipeline:
  Upload → OCR → Text Cleaning → Summarization → Risk Detection → Response
"""

import os
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from backend.config import Config

# Point HuggingFace cache to the project-local directory before any imports
os.environ.setdefault("HF_HOME", Config.HF_HOME)

from ocr.ocr_engine import OCREngine
from utils.text_cleaner import TextCleaner
from nlp.summarizer import LegalSummarizer
from models.risk_rules import RiskRuleEngine
from models.risk_classifier import RiskClassifier

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)

# ── All engine singletons are lazy — loaded on first request ─────────────────
# This ensures a misconfigured path or missing model never crashes app startup.
_ocr_engine: OCREngine | None = None
_text_cleaner: TextCleaner | None = None
_rule_engine: RiskRuleEngine | None = None
_summarizer: LegalSummarizer | None = None
_ml_engine: RiskClassifier | None = None


def _get_ocr_engine() -> OCREngine:
    """Return the shared OCREngine, creating it on first access."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initialising OCREngine with Tesseract at: %s", Config.TESSERACT_PATH)
        _ocr_engine = OCREngine(tesseract_path=Config.TESSERACT_PATH)
        logger.info("OCREngine ready.")
    return _ocr_engine


def _get_text_cleaner() -> TextCleaner:
    """Return the shared TextCleaner, creating it on first access."""
    global _text_cleaner
    if _text_cleaner is None:
        _text_cleaner = TextCleaner()
    return _text_cleaner


def _get_rule_engine() -> RiskRuleEngine:
    """Return the shared RiskRuleEngine, creating it on first access."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RiskRuleEngine()
    return _rule_engine


def _get_summarizer() -> LegalSummarizer:
    """Return the shared LegalSummarizer, creating it on first access."""
    global _summarizer
    if _summarizer is None:
        logger.info("Loading LegalSummarizer (first request)…")
        _summarizer = LegalSummarizer()
        logger.info("LegalSummarizer loaded.")
    return _summarizer


def _get_ml_engine() -> RiskClassifier:
    """Return the shared RiskClassifier, creating it on first access."""
    global _ml_engine
    if _ml_engine is None:
        logger.info("Loading RiskClassifier (first request)…")
        _ml_engine = RiskClassifier()
        logger.info("RiskClassifier loaded.")
    return _ml_engine


def _is_allowed_file(filename: str) -> bool:
    """Return True when the file extension is in the allowed set."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


@analysis_bp.route("/analyze", methods=["POST"])
def analyze_document():
    """
    POST /api/analyze

    Pipeline:
        Image → OCR → Text Cleaning → Summarization → Risk Detection
    """
    # ── 1. File presence check ──────────────────────────────────────────────
    if "document" not in request.files:
        return jsonify({"error": "No document uploaded. Include a 'document' field."}), 400

    file = request.files["document"]
    filename = secure_filename(file.filename or "")

    if not filename:
        return jsonify({"error": "Invalid or empty filename."}), 400

    # ── 2. Extension validation ─────────────────────────────────────────────
    if not _is_allowed_file(filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(Config.ALLOWED_EXTENSIONS))}"
        }), 415

    # ── 3. Save file ────────────────────────────────────────────────────────
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    try:
        file.save(file_path)
        logger.info("Document saved: %s", file_path)
    except OSError as exc:
        logger.error("Failed to save file %s: %s", filename, exc)
        return jsonify({"error": "Could not save uploaded file."}), 500

    # ── 4. OCR ──────────────────────────────────────────────────────────────
    try:
        ocr_result = _get_ocr_engine().extract_text(file_path)
    except Exception as exc:
        logger.error("OCR failed for %s: %s", filename, exc)
        return jsonify({"error": "OCR processing failed.", "detail": str(exc)}), 500

    if ocr_result["status"] == "low_confidence":
        logger.warning("Low OCR confidence (%.2f) for %s", ocr_result["confidence"], filename)
        return jsonify({
            "warning": "Low OCR confidence — result may be inaccurate.",
            "confidence": ocr_result["confidence"],
        }), 422

    # ── 5. Text cleaning ────────────────────────────────────────────────────
    try:
        cleaner = _get_text_cleaner()
        cleaned_text = cleaner.clean_text(ocr_result["text"])
        clauses = cleaner.split_into_clauses(cleaned_text)
        logger.debug("Text cleaned. Clause count: %d", len(clauses))
    except Exception as exc:
        logger.error("Text cleaning failed: %s", exc)
        return jsonify({"error": "Text cleaning failed.", "detail": str(exc)}), 500

    # ── 6. Summarisation ────────────────────────────────────────────────────
    try:
        summary = _get_summarizer().summarize(cleaned_text)
        logger.debug("Summarization complete.")
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return jsonify({"error": "Summarization failed.", "detail": str(exc)}), 500

    # ── 7. Risk detection ───────────────────────────────────────────────────
    try:
        rule_risks = _get_rule_engine().analyze_clauses(clauses)
        ml_risks = _get_ml_engine().analyze_clauses(clauses)
        logger.debug("Risk analysis complete. Rule: %d, ML: %d", len(rule_risks), len(ml_risks))
    except Exception as exc:
        logger.error("Risk analysis failed: %s", exc)
        return jsonify({"error": "Risk analysis failed.", "detail": str(exc)}), 500

    # ── 8. Response ─────────────────────────────────────────────────────────
    return jsonify({
        "summary": summary,
        "ocr_confidence": ocr_result["confidence"],
        "ocr_metrics": {
            "confidence":  ocr_result["confidence"],
            "word_count":  ocr_result["word_count"],
            "char_count":  ocr_result["char_count"],
            "token_count": ocr_result["token_count"],
            "status":      ocr_result["status"],
        },
        "risk_analysis": {
            "rule_based": rule_risks,
            "ml_based":   ml_risks,
        },
    }), 200
