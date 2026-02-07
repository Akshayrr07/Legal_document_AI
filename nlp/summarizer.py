from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import math
import os
from dotenv import load_dotenv

load_dotenv()


class LegalSummarizer:
    """
    Summarization engine for long legal documents.
    Uses chunking strategy to handle token limits safely.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_chunk_words: int = 450,
        summary_max_length: int = 150,
        summary_min_length: int = 60
    ):
        self.max_chunk_words = max_chunk_words
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hf_token = os.getenv("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token).to(self.device)
        
        # Create new generation config without problematic settings
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.max_length = self.summary_max_length
        gen_config.min_length = self.summary_min_length
        gen_config.num_beams = 4
        gen_config.early_stopping = True
        self.model.generation_config = gen_config
        
        # Configure generation parameters
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id

    def summarize(self, text: str) -> str:
        """
        Generate a plain-language summary for a legal document.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text for summarization.")

        chunks = self._chunk_text(text)

        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            summary_ids = self.model.generate(inputs["input_ids"])

            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)

        return self._merge_summaries(summaries)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split long legal text into word-based chunks.
        """
        words = text.split()
        total_words = len(words)

        num_chunks = math.ceil(total_words / self.max_chunk_words)
        chunks = []

        for i in range(num_chunks):
            start = i * self.max_chunk_words
            end = start + self.max_chunk_words
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _merge_summaries(summaries: List[str]) -> str:
        """
        Merge chunk summaries into a coherent final summary.
        """
        if not summaries:
            return ""

        if len(summaries) == 1:
            return summaries[0]

        merged = " ".join(summaries)
        return merged.strip()
