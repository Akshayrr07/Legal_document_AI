from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

load_dotenv()


class RiskClassifier:
    """
    Lightweight ML-based risk classifier using transformer embeddings.
    """

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        
        hf_token = os.getenv("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token)
        self.model.eval()

        # Risk thresholds (calibrated heuristically)
        self.thresholds = {
            "High": 0.75,
            "Medium": 0.55,
            "Low": 0.40
        }

    def _embed(self, text: str) -> np.ndarray:
        """
        Generate sentence embedding using mean pooling.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def analyze_clauses(self, clauses: List[str]) -> List[Dict]:
        """
        Assign risk levels to clauses based on semantic signals.
        """
        results = []

        for clause in clauses:
            embedding = self._embed(clause)
            score = float(np.linalg.norm(embedding) / 10)  # normalized heuristic

            if score >= self.thresholds["High"]:
                level = "High"
            elif score >= self.thresholds["Medium"]:
                level = "Medium"
            elif score >= self.thresholds["Low"]:
                level = "Low"
            else:
                continue  # ignore negligible risk

            results.append({
                "clause": clause,
                "risk_level": level,
                "confidence_score": round(score, 2),
                "source": "ml_based"
            })

        return results
