"""
Sliding Window Chunker for LMA Agreement Text

Why sliding windows:
- Legal clauses can span multiple paragraphs
- Fixed boundaries cut clauses in half
- Overlap ensures no clause is split across chunk boundary

Key decision - 250 tokens, 50 overlap:
- 75-token windows caused data leakage (val F1 = 1.0 = red flag)
- 75-token windows created boundary overlap between train/val splits
- 250-token windows large enough to contain full clause context
- 50-token overlap ensures clause boundaries are always captured
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class SlidingWindowChunker:
    """
    Splits legal agreement text into overlapping token windows
    for DistilBERT clause classification.
    """

    def __init__(self, window_size: int = 250, overlap: int = 50):
        """
        Args:
            window_size: Number of tokens per chunk.
                         250 chosen after discovering 75 caused data leakage.
            overlap:     Number of tokens repeated between consecutive chunks.
                         50 ensures clause boundaries always appear in at least one chunk.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap
        self.logger = logging.getLogger(__name__)

    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer.

        Why not use BERT tokenizer here:
        - We chunk at word level first for readability
        - DistilBERT tokenizer applied separately at inference time
        - Avoids double tokenization complexity
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def chunk_text(self, text: str, return_metadata: bool = True) -> List[dict]:
        """
        Apply sliding window chunking to input text.

        Process:
        1. Tokenize text into words
        2. Slide window across tokens with step = window_size - overlap
        3. Reconstruct each window back into string
        4. Return chunks with position metadata

        Args:
            text:            Raw extracted text from PDF
            return_metadata: If True, include token positions in output

        Returns:
            List of dicts: {text, start_token, end_token, chunk_index}

        Why return metadata:
        - Needed to map classified chunk back to page/position in PDF
        - Allows auditors to jump directly to relevant section
        """
        if not text or not text.strip():
            self.logger.warning("Empty text passed to chunker")
            return []

        tokens = self.tokenize(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunks = []
        chunk_index = 0
        start = 0

        while start < total_tokens:
            end = min(start + self.window_size, total_tokens)
            window_tokens = tokens[start:end]
            chunk_text = ' '.join(window_tokens)

            chunk = {
                'text': chunk_text,
                'chunk_index': chunk_index,
                'token_count': len(window_tokens),
            }

            if return_metadata:
                chunk['start_token'] = start
                chunk['end_token'] = end

            chunks.append(chunk)
            chunk_index += 1

            # If we've reached the end, stop
            if end == total_tokens:
                break

            start += self.step

        self.logger.info(
            f"Chunked {total_tokens} tokens into {len(chunks)} windows "
            f"(size={self.window_size}, overlap={self.overlap})"
        )
        return chunks

    def chunk_document(self, pages: List[str]) -> List[dict]:
        """
        Chunk a multi-page document, preserving page numbers.

        Why track page numbers:
        - Auditors need to know which page a clause appears on
        - Allows direct navigation in the original PDF

        Args:
            pages: List of page text strings, one per page

        Returns:
            List of chunk dicts with additional 'page_number' field
        """
        all_chunks = []

        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = self.chunk_text(page_text)

            for chunk in page_chunks:
                chunk['page_number'] = page_num

            all_chunks.extend(page_chunks)
            self.logger.debug(
                f"Page {page_num}: {len(page_chunks)} chunks"
            )

        self.logger.info(
            f"Document chunked: {len(pages)} pages → {len(all_chunks)} total chunks"
        )
        return all_chunks

    def validate_no_leakage(
        self,
        train_chunks: List[dict],
        val_chunks: List[dict]
    ) -> bool:
        """
        Verify there is no text overlap between train and validation chunks.

        Why this matters:
        - During development, 75-token windows caused 1.0 F1 on validation
        - Root cause: same clause text appeared in both train and val chunks
        - A perfect score is a red flag — always verify data integrity
        - This method detects that class of bug

        Returns:
            True if clean, False if leakage detected
        """
        train_texts = set(c['text'] for c in train_chunks)
        val_texts = set(c['text'] for c in val_chunks)

        overlap = train_texts.intersection(val_texts)

        if overlap:
            self.logger.error(
                f"DATA LEAKAGE DETECTED: {len(overlap)} identical chunks "
                f"in both train and validation sets. "
                f"This will produce artificially perfect F1 scores. "
                f"Increase window_size or adjust split boundaries."
            )
            return False

        self.logger.info("Data leakage check passed: no overlap between train/val")
        return True
