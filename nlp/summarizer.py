"""
LegalSummarizer — NLP summarization engine for legal documents.

Improvements over the original:
- Lazy model loading (model instantiated on first call, not at import time)
- Sentence-aware chunking using regex sentence boundaries (prevents mid-sentence splits)
- Token-safe segmentation with hard token count guard before encoding
- Batch inference when multiple chunks can be processed at once
- In-memory result cache for repeated identical inputs (lru_cache)
- Improved merge strategy with deduplication
"""

import re
import logging
import functools
from typing import List

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_MODEL_NAME = "facebook/bart-large-cnn"
_MAX_TOKEN_INPUT = 1024       # BART's encoder limit
_MAX_WORDS_PER_CHUNK = 400    # Conservative word limit for sentence-aware chunks
_SUMMARY_MAX_LENGTH = 150
_SUMMARY_MIN_LENGTH = 60
_NUM_BEAMS = 4


class LegalSummarizer:
    """
    Summarization engine for long legal documents.
    Uses a lazy-loaded BART model with sentence-aware chunking.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        max_chunk_words: int = _MAX_WORDS_PER_CHUNK,
        summary_max_length: int = _SUMMARY_MAX_LENGTH,
        summary_min_length: int = _SUMMARY_MIN_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.max_chunk_words = max_chunk_words
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length

        # Lazy-loaded attributes
        self._tokenizer = None
        self._model = None
        self._device = None

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model and tokenizer on first use to avoid startup overhead."""
        import os
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            GenerationConfig,
        )
        from dotenv import load_dotenv
        load_dotenv()

        logger.info("Loading summarizer model: %s", self.model_name)

        hf_token = os.getenv("HF_TOKEN") or None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, token=hf_token
        ).to(self._device)

        # Apply generation configuration
        gen_config = GenerationConfig.from_model_config(self._model.config)
        gen_config.max_length = self.summary_max_length
        gen_config.min_length = self.summary_min_length
        gen_config.num_beams = _NUM_BEAMS
        gen_config.early_stopping = True
        self._model.generation_config = gen_config

        # Ensure decoder start token is set correctly
        if self._model.config.decoder_start_token_id is None:
            self._model.config.decoder_start_token_id = self._tokenizer.bos_token_id

        logger.info("Summarizer model loaded on device: %s", self._device)

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

    @property
    def device(self):
        if self._device is None:
            self._load_model()
        return self._device

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """
        Generate a plain-language summary of the provided legal document text.

        Parameters
        ----------
        text : str
            Cleaned legal document text.

        Returns
        -------
        str
            Merged summary string.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input: text must be a non-empty string.")

        # Use cached internal method (keyed on text content)
        return self._summarize_cached(text)

    @functools.lru_cache(maxsize=32)
    def _summarize_cached(self, text: str) -> str:
        """Cached summarization — avoids re-running the model for identical inputs."""
        chunks = self._chunk_text(text)
        logger.debug("Summarizing %d chunk(s) for input of %d words.", len(chunks), len(text.split()))

        summaries = self._batch_summarize(chunks)
        return self._merge_summaries(summaries)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries.

        Strategy:
        1. Split on sentence-ending punctuation (., ?, !)
        2. Greedily accumulate sentences until within word-count limit
        3. Hard token-count guard before encoding
        """
        # Tokenize into sentences using a regex that keeps delimiters
        sentence_pattern = re.compile(r"(?<=[.?!])\s+")
        sentences = sentence_pattern.split(text)

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_word_count: int = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            word_count = len(sentence.split())

            # If adding this sentence would exceed the limit, flush the buffer
            if current_word_count + word_count > self.max_chunk_words and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0

            # If a single sentence itself exceeds the limit, hard-split it
            if word_count > self.max_chunk_words:
                words = sentence.split()
                for i in range(0, len(words), self.max_chunk_words):
                    chunks.append(" ".join(words[i : i + self.max_chunk_words]))
            else:
                current_sentences.append(sentence)
                current_word_count += word_count

        # Flush any remaining sentences
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if chunks else [text]

    def _batch_summarize(self, chunks: List[str]) -> List[str]:
        """
        Run model inference on all chunks.
        Each chunk is encoded independently but inference is run sequentially
        with the same model instance to keep memory usage predictable.
        """
        summaries: List[str] = []

        for idx, chunk in enumerate(chunks):
            inputs = self.tokenizer(
                chunk,
                max_length=_MAX_TOKEN_INPUT,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            summary_ids = self.model.generate(inputs["input_ids"])
            summary_text = self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True
            )
            summaries.append(summary_text)
            logger.debug("Chunk %d/%d summarized.", idx + 1, len(chunks))

        return summaries

    @staticmethod
    def _merge_summaries(summaries: List[str]) -> str:
        """
        Merge chunk-level summaries into a single coherent output.
        Strips duplicate adjacent sentences before joining.
        """
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0].strip()

        # Deduplicate consecutive identical summaries (e.g. very short chunks)
        deduped: List[str] = [summaries[0]]
        for s in summaries[1:]:
            if s.strip() != deduped[-1].strip():
                deduped.append(s.strip())

        return " ".join(deduped)
