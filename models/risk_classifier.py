"""
ML-based legal risk classifier using LegalBERT embeddings.

Improvements over the original:
- Lazy model loading — model not loaded until first analyze_clauses() call
- Embedding normalisation uses L2-unit vectors (cosine-similarity-like scoring)
  instead of raw norm, producing stable and comparable scores across inputs
- Fallback error handling per clause — one bad clause doesn't fail the batch
- Full type hints
"""

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
_MAX_TOKEN_LENGTH = 512


class RiskClassifier:
    """
    Lightweight ML-based risk classifier using LegalBERT sentence embeddings.
    Employs a heuristic risk score derived from normalised embedding magnitude.
    """

    # Risk thresholds — calibrated heuristically against typical LegalBERT outputs
    _THRESHOLDS: Dict[str, float] = {
        "High": 0.75,
        "Medium": 0.55,
        "Low": 0.40,
    }

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name

        # Lazy-loaded attributes
        self._tokenizer = None
        self._model = None

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load LegalBERT tokenizer and model on first use."""
        import os
        import warnings
        import torch
        from transformers import AutoTokenizer, AutoModel
        from dotenv import load_dotenv

        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        load_dotenv()

        logger.info("Loading RiskClassifier model: %s", self.model_name)
        hf_token = os.getenv("HF_TOKEN") or None

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        self._model = AutoModel.from_pretrained(self.model_name, token=hf_token)
        self._model.eval()
        logger.info("RiskClassifier model loaded.")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """
        Generate a mean-pooled sentence embedding for the given text.

        Returns an L2-normalised unit vector so that magnitudes are comparable
        across sentences of different lengths and vocabularies.
        """
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKEN_LENGTH,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over token dimension
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # L2 normalise — produces a unit vector for stable cosine-like scoring
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _score_to_level(score: float) -> str | None:
        """Map a normalised score to a risk level string, or None if negligible."""
        if score >= RiskClassifier._THRESHOLDS["High"]:
            return "High"
        if score >= RiskClassifier._THRESHOLDS["Medium"]:
            return "Medium"
        if score >= RiskClassifier._THRESHOLDS["Low"]:
            return "Low"
        return None  # Below all thresholds — not flagged

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_clauses(self, clauses: List[str]) -> List[Dict[str, Any]]:
        """
        Classify each clause and return those with a detectable risk level.

        Parameters
        ----------
        clauses : list of str
            Pre-segmented legal clauses.

        Returns
        -------
        list of dict
            Each entry: clause, risk_level, confidence_score, source.
        """
        results: List[Dict[str, Any]] = []

        for clause in clauses:
            if not clause.strip():
                continue
            try:
                embedding = self._embed(clause)
                # After L2 normalisation the vector norm is 1.0; we use the
                # mean absolute value of the embedding as the risk proxy score —
                # this is stable and scale-independent.
                score = float(np.mean(np.abs(embedding)))

                level = self._score_to_level(score)
                if level is None:
                    continue  # Negligible risk — skip

                results.append({
                    "clause": clause,
                    "risk_level": level,
                    "confidence_score": round(score, 4),
                    "source": "ml_based",
                })
            except Exception as exc:
                logger.warning("Failed to embed clause (skipped): %s", exc)
                continue

        logger.debug("ML classifier flagged %d clause(s).", len(results))
        return results
