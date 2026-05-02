"""
FastAPI Endpoint for LMA Clause Identification Tool

Exposes the full pipeline as a REST API:
PDF upload → chunk → classify → threshold → SBERT match → JSON response
"""

import logging
import time
from pathlib import Path
from typing import List, Optional
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.data_processing.pdf_extractor import PDFExtractor
from src.data_processing.chunker import SlidingWindowChunker
from src.models.classifier import LMAClassifier
from src.models.threshold import ConfidenceThresholder
from src.models.embedder import SBERTEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LMA Clause Identification Tool",
    description=(
        "NLP pipeline for identifying and classifying legal clauses "
        "in Loan Market Association agreements. "
        "Built at Standard Chartered Global Business Services."
    ),
    version="1.0.0",
)

# ─── Response Models ──────────────────────────────────────────────────────────

class ClauseMatch(BaseModel):
    text: str
    similarity: float
    rank: int

class IdentifiedClause(BaseModel):
    clause_type: str
    confidence: float
    chunk_text: str
    page_number: Optional[int]
    chunk_index: int
    threshold_status: str
    gold_standard_matches: List[ClauseMatch]

class AnalysisResponse(BaseModel):
    document_name: str
    total_chunks: int
    clauses_identified: int
    uncertain_count: int
    processing_time_seconds: float
    clauses: List[IdentifiedClause]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    threshold: float

# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    classifier: Optional[LMAClassifier] = None
    thresholder: Optional[ConfidenceThresholder] = None
    embedder: Optional[SBERTEmbedder] = None
    extractor: PDFExtractor = PDFExtractor()
    chunker: SlidingWindowChunker = SlidingWindowChunker(
        window_size=250, overlap=50
    )

state = AppState()


@app.on_event("startup")
async def load_models():
    """
    Load all models at startup.

    Why load at startup rather than per-request:
    - Models are large (300MB+), loading per-request adds 5-10s latency
    - Multiple concurrent requests would each load their own copy
    - Single shared model instance with model.eval() is thread-safe for inference
    """
    logger.info("Loading models...")

    model_dir = Path("models")

    # Load classifier
    classifier_path = model_dir / "distilbert_lma"
    if classifier_path.exists():
        state.classifier = LMAClassifier.load(str(classifier_path))
        logger.info("✓ DistilBERT classifier loaded")
    else:
        logger.warning(
            f"Classifier not found at {classifier_path}. "
            f"Run fine-tuning first or place model files in models/distilbert_lma/"
        )

    # Load threshold
    threshold_path = model_dir / "threshold.json"
    if threshold_path.exists():
        state.thresholder = ConfidenceThresholder.load(str(threshold_path))
        logger.info(f"✓ Threshold loaded: {state.thresholder.threshold:.3f}")
    else:
        state.thresholder = ConfidenceThresholder(threshold=0.72)
        logger.info("✓ Using default threshold: 0.72")

    # Load SBERT embedder + FAISS index
    embedder_path = model_dir / "sbert_legal"
    index_path = model_dir / "faiss_index"
    if embedder_path.exists() and index_path.exists():
        state.embedder = SBERTEmbedder(model_name=str(embedder_path))
        state.embedder.load_index(str(index_path))
        logger.info("✓ SBERT embedder + FAISS index loaded")
    else:
        logger.warning(
            "SBERT embedder or FAISS index not found. "
            "Gold-standard matching will be unavailable."
        )

    logger.info("Model loading complete.")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Used by Kubernetes liveness and readiness probes.
    Returns 200 only when models are loaded and ready.
    """
    models_loaded = state.classifier is not None
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        threshold=state.thresholder.threshold if state.thresholder else 0.0,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze a PDF loan agreement for LMA clauses.

    Process:
    1. Save uploaded PDF to temp file
    2. Extract text page by page
    3. Chunk into 250-token sliding windows
    4. Classify each chunk with DistilBERT
    5. Filter by confidence threshold
    6. For each accepted clause, find SBERT gold-standard matches
    7. Return structured JSON

    Args:
        file: PDF file upload (multipart/form-data)

    Returns:
        AnalysisResponse with all identified clauses, confidence scores,
        and gold-standard matches

    Example curl:
        curl -X POST http://localhost:8000/analyze \\
             -F "file=@loan_agreement.pdf"
    """
    if state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not loaded. Check server logs."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    start_time = time.time()

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text
        pages = state.extractor.extract_pages(tmp_path)

        # Chunk
        chunks = state.chunker.chunk_document(pages)
        logger.info(f"Extracted {len(chunks)} chunks from {len(pages)} pages")

        # Classify
        predictions = state.classifier.predict(chunks)

        # Threshold
        accepted, uncertain = state.thresholder.filter(predictions)
        logger.info(
            f"After thresholding: {len(accepted)} accepted, "
            f"{len(uncertain)} uncertain"
        )

        # Build response
        clause_results = []

        for pred in accepted:
            # SBERT gold-standard matching
            matches = []
            if state.embedder is not None:
                try:
                    raw_matches = state.embedder.search(pred['text'], top_k=3)
                    matches = [
                        ClauseMatch(
                            text=m['text'],
                            similarity=m['similarity'],
                            rank=m['rank'],
                        )
                        for m in raw_matches
                    ]
                except Exception as e:
                    logger.warning(f"SBERT search failed: {e}")

            clause_results.append(
                IdentifiedClause(
                    clause_type=pred['predicted_label'],
                    confidence=round(pred['confidence'], 4),
                    chunk_text=pred['text'],
                    page_number=pred.get('page_number'),
                    chunk_index=pred['chunk_index'],
                    threshold_status=pred['threshold_status'],
                    gold_standard_matches=matches,
                )
            )

        processing_time = round(time.time() - start_time, 2)

        return AnalysisResponse(
            document_name=file.filename,
            total_chunks=len(chunks),
            clauses_identified=len(accepted),
            uncertain_count=len(uncertain),
            processing_time_seconds=processing_time,
            clauses=clause_results,
        )

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
