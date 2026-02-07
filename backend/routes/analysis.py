import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Set HuggingFace cache to avoid repeated downloads
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../../.cache/huggingface")

from ocr.ocr_engine import OCREngine
from utils.text_cleaner import TextCleaner
from nlp.summarizer import LegalSummarizer
from models.risk_rules import RiskRuleEngine
from models.risk_classifier import RiskClassifier

load_dotenv()

analysis_bp = Blueprint("analysis", __name__)

# Initialize engines once (performance-safe)
ocr_engine = OCREngine(
    tesseract_path=os.getenv("TESSERACT_PATH")
)
text_cleaner = TextCleaner()
summarizer = LegalSummarizer()
rule_engine = RiskRuleEngine()
ml_engine = RiskClassifier()


@analysis_bp.route("/analyze", methods=["POST"])
def analyze_document():
    """
    Orchestrates:
    Image → OCR → Cleaning → Summarization → Risk Analysis
    """
    if "document" not in request.files:
        return jsonify({"error": "No document uploaded"}), 400

    file = request.files["document"]
    filename = secure_filename(file.filename)

    if filename == "":
        return jsonify({"error": "Invalid filename"}), 400

    file_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"], filename
    )
    file.save(file_path)

    # ---- OCR ----
    ocr_result = ocr_engine.extract_text(file_path)

    if ocr_result["status"] == "low_confidence":
        return jsonify({
            "warning": "Low OCR confidence",
            "confidence": ocr_result["confidence"]
        }), 422

    # ---- Text Cleaning ----
    cleaned_text = text_cleaner.clean_text(ocr_result["text"])
    clauses = text_cleaner.split_into_clauses(cleaned_text)

    # ---- Summarization ----
    summary = summarizer.summarize(cleaned_text)

    # ---- Risk Analysis ----
    rule_risks = rule_engine.analyze_clauses(clauses)
    ml_risks = ml_engine.analyze_clauses(clauses)

    response = {
        "summary": summary,
        "ocr_confidence": ocr_result["confidence"],
        "risk_analysis": {
            "rule_based": rule_risks,
            "ml_based": ml_risks
        }
    }

    return jsonify(response), 200
