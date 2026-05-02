"""
PDF Text Extractor for LMA Loan Agreements

Handles extraction of raw text from multi-page PDF agreements
for downstream chunking and NLP classification.

Why pdfplumber over PyMuPDF:
- Better handling of complex table structures in legal docs
- More accurate character position data
- Preserves line breaks which are semantically meaningful in contracts
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")


class PDFExtractor:
    """
    Extracts text from LMA agreement PDFs page by page.
    Preserves paragraph structure for downstream chunking.
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_pages(self, pdf_path: str) -> List[str]:
        """
        Extract text from each page of a PDF.

        Process:
        1. Open PDF with pdfplumber
        2. Extract text from each page
        3. Clean whitespace artifacts
        4. Return list of page strings

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of strings, one per page.
            Empty string for pages where extraction fails.

        Why page-level extraction:
        - Preserves document structure (sections, clauses per page)
        - Allows page-number attribution in audit output
        - Smaller units = better error isolation per page
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required. Install with: pip install pdfplumber"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                self.logger.info(
                    f"Extracting {total_pages} pages from: {pdf_path.name}"
                )

                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text:
                            # Normalize whitespace while preserving line structure
                            text = self._clean_text(text)
                            pages.append(text)
                        else:
                            self.logger.warning(
                                f"Page {page_num}: no text extracted (may be image-based)"
                            )
                            pages.append("")

                    except Exception as e:
                        self.logger.error(f"Page {page_num} extraction failed: {e}")
                        pages.append("")

        except Exception as e:
            self.logger.error(f"PDF open failed: {e}")
            raise

        non_empty = sum(1 for p in pages if p.strip())
        self.logger.info(
            f"Extraction complete: {non_empty}/{len(pages)} pages with text"
        )
        return pages

    def extract_full_text(self, pdf_path: str) -> str:
        """
        Extract full document text as a single string.

        Used when page boundaries are not important for classification.
        Page separator (\f) preserved for optional downstream splitting.
        """
        pages = self.extract_pages(pdf_path)
        return "\f".join(pages)

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text without losing clause structure.

        Why careful cleaning:
        - Legal clauses often span multiple lines
        - Removing all line breaks loses paragraph boundaries
        - Only collapse excess whitespace, keep single newlines
        """
        import re
        # Remove null bytes and control characters except newlines/tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple spaces to single
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse more than 2 newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
