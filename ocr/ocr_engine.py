import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
from typing import Dict


class OCREngine:
    """
    OCR Engine for extracting text from scanned legal documents.
    Uses Tesseract OCR with image preprocessing for improved accuracy.
    """

    def __init__(self, tesseract_path: str):
        if not os.path.exists(tesseract_path):
            raise FileNotFoundError(
                f"Tesseract not found at path: {tesseract_path}"
            )
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    @staticmethod
    def _preprocess_image(image_path: str) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy:
        - Grayscale
        - Noise removal
        - Thresholding
        """
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Invalid image file or unreadable image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    @staticmethod
    def _calculate_confidence(ocr_data: Dict) -> float:
        """
        Calculate average OCR confidence score.
        """
        confidences = [
            conf
            for conf in ocr_data["conf"]
            if isinstance(conf, int) and conf >= 0
        ]

        if not confidences:
            return 0.0

        return round(sum(confidences) / len(confidences), 2)

    def extract_text(self, image_path: str) -> Dict:
        """
        Perform OCR on the given image.

        Returns:
        {
            "text": extracted_text,
            "confidence": average_confidence,
            "status": "success" | "low_confidence"
        }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        processed_image = self._preprocess_image(image_path)

        custom_config = r"--oem 3 --psm 6"
        ocr_data = pytesseract.image_to_data(
            processed_image,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        extracted_text = " ".join(
            [word for word in ocr_data["text"] if word.strip()]
        )

        confidence = self._calculate_confidence(ocr_data)

        status = "success" if confidence >= 60 else "low_confidence"

        return {
            "text": extracted_text.strip(),
            "confidence": confidence,
            "status": status
        }
