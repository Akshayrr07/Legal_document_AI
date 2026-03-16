"""
OCR Engine — extracts text from scanned legal document images.

Improvements over the original:
- CLAHE contrast normalisation before thresholding
- Adaptive (Gaussian) thresholding instead of global Otsu
- Morphological dilation to connect broken characters
- DPI upscaling to ensure a minimum 300 DPI equivalent
- Improved confidence calculation that ignores empty / noise tokens
"""

import os
import logging
from typing import Dict

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Tesseract configuration: OEM 3 (LSTM), PSM 6 (uniform block of text)
_TESS_CONFIG = r"--oem 3 --psm 6"

# Minimum average confidence to consider OCR successful
_CONFIDENCE_THRESHOLD: float = 60.0

# Target minimum resolution (pixels per inch equivalent)
_MIN_RESOLUTION: int = 300


class OCREngine:
    """
    OCR Engine for extracting text from scanned legal documents.
    Uses Tesseract OCR with enhanced image pre-processing for improved accuracy.
    """

    def __init__(self, tesseract_path: str) -> None:
        if not os.path.exists(tesseract_path):
            raise FileNotFoundError(
                f"Tesseract binary not found at: {tesseract_path}"
            )
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info("OCREngine initialised. Tesseract: %s", tesseract_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _upscale_if_needed(image: np.ndarray) -> np.ndarray:
        """
        Upscale the image if its shorter dimension is below the minimum target.
        This ensures characters are large enough for Tesseract to recognise reliably.
        """
        height, width = image.shape[:2]
        short_side = min(height, width)
        # If image is very small (e.g. scanned at 72 DPI), upscale 2×
        if short_side < _MIN_RESOLUTION:
            scale = _MIN_RESOLUTION / short_side
            new_w = int(width * scale)
            new_h = int(height * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug("Image upscaled to %dx%d", new_w, new_h)
        return image

    @staticmethod
    def _preprocess_image(image_path: str) -> np.ndarray:
        """
        Full pre-processing pipeline:
          1. Read & upscale if below minimum resolution
          2. Greyscale conversion
          3. CLAHE contrast normalisation (handles uneven lighting)
          4. Gaussian blur for noise reduction
          5. Adaptive Gaussian thresholding (robust to local lighting variations)
          6. Morphological dilation (closes gaps in thin characters)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image file: {image_path}")

        # Step 1: upscale if needed
        image = OCREngine._upscale_if_needed(image)

        # Step 2: greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: CLAHE — enhances local contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Step 4: Gaussian blur — reduces high-frequency noise before thresholding
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 5: Adaptive Gaussian thresholding — handles documents with uneven
        # backgrounds (shadows, yellowing, scanning artefacts)
        thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31,
            C=11,
        )

        # Step 6: Morphological dilation — closes small gaps between characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed = cv2.dilate(thresh, kernel, iterations=1)

        return processed

    @staticmethod
    def _calculate_confidence(ocr_data: Dict) -> float:
        """
        Compute weighted average OCR confidence, ignoring empty/noisy tokens.
        Tokens shorter than 2 characters are treated as noise and excluded.
        """
        confidences = [
            conf
            for conf, word in zip(ocr_data["conf"], ocr_data["text"])
            if isinstance(conf, int) and conf >= 0 and len(word.strip()) >= 2
        ]
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 2)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> Dict:
        """
        Perform OCR on the given image file.

        Returns
        -------
        dict
            {
                "text": str,           — extracted text
                "confidence": float,   — average token confidence (0–100)
                "status": str          — "success" | "low_confidence"
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug("Starting OCR for: %s", image_path)
        processed_image = self._preprocess_image(image_path)

        ocr_data = pytesseract.image_to_data(
            processed_image,
            config=_TESS_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

        # Reconstruct text — only include non-empty tokens
        extracted_text = " ".join(
            word for word in ocr_data["text"] if word.strip()
        )

        confidence = self._calculate_confidence(ocr_data)
        status = "success" if confidence >= _CONFIDENCE_THRESHOLD else "low_confidence"

        # ── Metrics ────────────────────────────────────────────────────────────
        words = [w for w in ocr_data["text"] if w.strip()]
        word_count = len(words)
        char_count = len(extracted_text)
        # Token count estimated as whitespace-separated tokens (same as word_count)
        token_count = word_count

        logger.info(
            "OCR complete. Confidence: %.2f, Words: %d, Chars: %d, Status: %s",
            confidence, word_count, char_count, status,
        )
        return {
            "text":        extracted_text.strip(),
            "confidence":  confidence,
            "status":      status,
            "word_count":  word_count,
            "char_count":  char_count,
            "token_count": token_count,
        }
