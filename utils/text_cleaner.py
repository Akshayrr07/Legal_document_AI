import re
from typing import List


class TextCleaner:
    """
    Cleans and normalizes OCR-extracted legal text
    to improve downstream NLP performance.
    """

    def __init__(self):
        # Common OCR noise patterns
        self.noise_patterns = [
            r"\n+",                  # Multiple newlines
            r"\s{2,}",               # Multiple spaces
            r"_+",                   # Underscore noise
            r"\f",                   # Form feed
        ]

    def clean_text(self, text: str) -> str:
        """
        Perform full text cleaning pipeline.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text for cleaning.")

        text = self._remove_non_ascii(text)
        text = self._remove_noise(text)
        text = self._normalize_punctuation(text)
        text = self._normalize_whitespace(text)

        return text.strip()

    def split_into_clauses(self, text: str) -> List[str]:
        """
        Heuristic clause segmentation for legal documents.
        """
        if not text:
            return []

        clause_delimiters = r"(;|\.\s|\n\d+\.\s|\n[A-Z][a-z]+\s)"
        raw_clauses = re.split(clause_delimiters, text)

        clauses = [
            clause.strip()
            for clause in raw_clauses
            if clause and len(clause.strip()) > 30
        ]

        return clauses

    @staticmethod
    def _remove_non_ascii(text: str) -> str:
        return re.sub(r"[^\x00-\x7F]+", " ", text)

    def _remove_noise(self, text: str) -> str:
        for pattern in self.noise_patterns:
            text = re.sub(pattern, " ", text)
        return text

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r",+", ",", text)
        text = re.sub(r";+", ";", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text)
