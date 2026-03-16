"""
Text cleaning and clause segmentation utilities for OCR-extracted legal text.

Improvements over the original:
- More precise clause segmentation regex that handles numbered sections (1., 2.) and
  lettered clauses (a., b.) common in legal documents
- Minimum word threshold (not just character count) to filter fragments
- All private methods fully type-hinted
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Minimum word count for a text segment to count as a valid clause
_MIN_CLAUSE_WORDS = 8

# Minimum character count retained for backward compatibility
_MIN_CLAUSE_CHARS = 30


class TextCleaner:
    """
    Cleans and normalises OCR-extracted legal text to improve downstream
    NLP and risk-detection performance.
    """

    # Pre-compiled noise patterns applied sequentially during cleaning
    _NOISE_PATTERNS: List[tuple] = [
        (re.compile(r"\f"),          " "),    # Form feed → space
        (re.compile(r"\r\n|\r"),     "\n"),   # Normalise line endings
        (re.compile(r"\n{2,}"),      "\n"),   # Collapse multiple blank lines
        (re.compile(r"_+"),          " "),    # Underscores (blanks in forms)
        (re.compile(r"\s{2,}"),      " "),    # Multiple spaces → single space
    ]

    # Clause delimiter pattern — splits on:
    #   - semicolons
    #   - full-stop followed by whitespace (end of sentence)
    #   - numbered list items: "1. " or "12. "
    #   - lettered items: "(a) " or "(i) "
    #   - ALL-CAPS section headers such as "WHEREAS " or "NOW THEREFORE "
    _CLAUSE_DELIMITER = re.compile(
        r"(?:;|\.\s+|(?<!\w)\d{1,2}\.\s+|\([a-z]+\)\s+|(?<=[a-z])\s+(?=[A-Z]{3,}))"
    )

    def clean_text(self, text: str) -> str:
        """
        Run the full text-cleaning pipeline on raw OCR output.

        Steps
        -----
        1. Remove non-ASCII characters
        2. Normalise line endings and form feeds
        3. Remove underscore/blank noise
        4. Normalise punctuation (repeated dots, commas, semicolons)
        5. Collapse whitespace

        Parameters
        ----------
        text : str
            Raw OCR-extracted text.

        Returns
        -------
        str
            Cleaned, normalised text ready for NLP processing.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")

        text = self._remove_non_ascii(text)
        text = self._apply_noise_patterns(text)
        text = self._normalise_punctuation(text)
        text = self._normalise_whitespace(text)

        cleaned = text.strip()
        logger.debug("Text cleaned. Length: %d → %d chars.", len(text), len(cleaned))
        return cleaned

    def split_into_clauses(self, text: str) -> List[str]:
        """
        Segment cleaned legal text into individual clauses.

        Uses a regex that recognises common legal document delimiters:
        semicolons, sentence boundaries, numbered/lettered list items,
        and ALL-CAPS section headers.

        Fragments shorter than ``_MIN_CLAUSE_WORDS`` words or
        ``_MIN_CLAUSE_CHARS`` characters are discarded.

        Parameters
        ----------
        text : str
            Cleaned legal document text.

        Returns
        -------
        list of str
            List of extracted clause strings.
        """
        if not text:
            return []

        raw_segments = self._CLAUSE_DELIMITER.split(text)

        clauses = [
            seg.strip()
            for seg in raw_segments
            if (
                seg
                and len(seg.strip()) >= _MIN_CLAUSE_CHARS
                and len(seg.strip().split()) >= _MIN_CLAUSE_WORDS
            )
        ]

        logger.debug("Clause segmentation produced %d clause(s).", len(clauses))
        return clauses

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_non_ascii(text: str) -> str:
        """Strip characters outside the printable ASCII range."""
        return re.sub(r"[^\x00-\x7F]+", " ", text)

    def _apply_noise_patterns(self, text: str) -> str:
        """Apply all pre-compiled noise-removal patterns in order."""
        for pattern, replacement in self._NOISE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _normalise_punctuation(text: str) -> str:
        """Collapse repeated punctuation to a single instance."""
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r",+", ",", text)
        text = re.sub(r";+", ";", text)
        return text

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse any remaining whitespace sequences to a single space."""
        return re.sub(r"\s+", " ", text)
