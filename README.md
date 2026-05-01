# LMA Clause Identification Tool

> Production NLP pipeline for automated legal clause identification in Loan Market Association agreements. Built and deployed at Standard Chartered Global Business Services.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF9D00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Dataiku](https://img.shields.io/badge/Dataiku-DSS-2AB1AC?style=flat-square)](https://dataiku.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## The Problem

Standard Chartered's legal and compliance team manually reviewed 100+ page LMA (Loan Market Association) loan agreements to verify that required clauses — Sanctions, Default, Representations, Definitions, Arbitration — were present and compliant.

This process was:
- **Slow** — each agreement took hours of manual review
- **Inconsistent** — different reviewers caught different things
- **Unscalable** — volume of agreements was growing faster than reviewer capacity

---

## The Solution

An end-to-end NLP pipeline that:
1. Extracts text from PDF agreements and chunks it into overlapping windows
2. Classifies each chunk using a fine-tuned DistilBERT classifier
3. Filters low-confidence predictions using a data-driven confidence threshold
4. Retrieves gold-standard sample language for matched clauses using SBERT semantic search
5. Returns structured output for auditor review

**Result: 90% reduction in audit time. 98.6% recall on rare clauses.**

---

## Architecture

```
PDF Agreement (100+ pages)
        │
        ▼
┌─────────────────────────────┐
│   ETL / Preprocessing       │
│  • pdfplumber text extract  │
│  • 250-token sliding windows│
│  • 50-token overlap         │
│  • Text normalisation       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   DistilBERT Classifier     │
│  • Fine-tuned on LMA data   │
│  • WeightedTrainer (20x)    │
│  • Outputs: class + logits  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Confidence Thresholding   │
│  • Minimum correct confid.  │
│  • Filters 95%+ false pos.  │
│  • Flags uncertain chunks   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   SBERT Semantic Search     │
│  • all-MiniLM-L6-v2         │
│  • Fine-tuned on legal pairs│
│  • Cosine similarity match  │
│  • Returns gold-standard    │
└──────────────┬──────────────┘
               │
               ▼
    Structured Audit Output
    (clause type + confidence
     + matched standard text)
```

---

## Key Technical Decisions

### Why DistilBERT over BERT?
DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its language understanding. For a production pipeline processing hundreds of documents, inference speed matters. The accuracy trade-off is negligible for classification tasks.

### Why 250-token windows with 50-token overlap?
During development, 75-token windows produced a 1.0 F1-score on validation — which is a red flag, not a success. Investigation revealed **data leakage**: clause text was appearing in both training and validation chunks due to the small window size causing boundary overlap. Increasing to 250-token windows with careful train-val splits eliminated the leakage and produced honest metrics (0.93 F1).

**Lesson: A perfect score should increase suspicion, not celebration.**

### Why WeightedTrainer with 20x class boost?
Over 90% of text in a loan agreement is irrelevant (boilerplate, formatting, recitals). Without class weighting, the model learns to predict "irrelevant" for everything and achieves high accuracy while missing every actual clause. The 20x class weight forces the model to pay attention to the rare positive examples.

### Why Minimum Correct Confidence for thresholding?
Two approaches were evaluated:
1. **KDE Plot method** — fit a kernel density to correct and incorrect prediction distributions, find the intersection
2. **Minimum Correct Confidence method** — find the lowest confidence score among all correct predictions

The KDE approach requires a separable incorrect prediction distribution. Since the fine-tuned model had very few incorrect predictions, no meaningful incorrect curve existed. The minimum correct confidence method was data-driven and robust.

### Why fine-tune SBERT for semantic search?
A generic SBERT model understands general language similarity. But legal language has domain-specific synonyms: "obligor shall comply with sanctions regimes" and "borrower must adhere to applicable restrictive measures" mean the same thing but share no words. Fine-tuning with `MultipleNegativesRankingLoss` on anchor-positive legal pairs teaches the model that these are semantically identical.

---

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 0.96 |
| Validation Accuracy | 0.93 |
| Weighted F1 | 0.93 |
| Recall on rare clauses | 98.6% |
| False positives filtered | 95%+ |
| Audit time reduction | 90% |

---

## Project Structure

```
lma-clause-identifier/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── model_config.yaml        # Thresholds, model paths, hyperparameters
├── src/
│   ├── data_processing/
│   │   ├── pdf_extractor.py     # PDF → raw text
│   │   ├── chunker.py           # Sliding window chunking
│   │   └── preprocessor.py     # Text cleaning and normalisation
│   ├── models/
│   │   ├── classifier.py        # DistilBERT fine-tuning + inference
│   │   ├── embedder.py          # SBERT embedding + semantic search
│   │   └── threshold.py        # Confidence threshold calibration
│   ├── api/
│   │   └── main.py             # FastAPI endpoint
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_chunker.py
│   ├── test_classifier.py
│   └── test_api.py
├── notebooks/
│   ├── 01_eda_and_chunking.ipynb
│   ├── 02_distilbert_finetuning.ipynb
│   ├── 03_threshold_calibration.ipynb
│   └── 04_sbert_finetuning.ipynb
└── Dockerfile
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/harshalbhambhani/lma-clause-identifier
cd lma-clause-identifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (requires DVC + AWS S3 access)
dvc pull
```

---

## Requirements

```
torch==1.13.1
transformers==4.26.0
sentence-transformers==2.2.2
datasets==2.10.1
scikit-learn==1.2.1
pandas==1.5.3
numpy==1.24.2
pdfplumber==0.9.0
fastapi==0.95.0
uvicorn==0.21.1
pydantic==1.10.7
```

---

## Usage

```python
from src.models.classifier import LMAClassifier
from src.data_processing.chunker import sliding_window_chunk

# Load the fine-tuned classifier
classifier = LMAClassifier.load("models/distilbert_finetuned/")

# Chunk a document
with open("agreement.pdf", "rb") as f:
    chunks = sliding_window_chunk(f, window_size=250, overlap=50)

# Classify chunks
results = classifier.predict(chunks, confidence_threshold=0.72)

# Print identified clauses
for result in results:
    print(f"Clause: {result.clause_type}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Text: {result.chunk_text[:100]}...")
    print()
```

---

## API

```bash
# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Analyze a document
curl -X POST http://localhost:8000/analyze \
  -F "file=@loan_agreement.pdf"

# Response
{
  "document_name": "loan_agreement.pdf",
  "clauses_found": [
    {
      "clause_type": "Sanctions",
      "confidence": 0.94,
      "chunk_text": "The Obligor shall comply with all applicable sanctions...",
      "matched_standard": "Standard sanctions compliance language..."
    }
  ],
  "processing_time_seconds": 2.3
}
```

---

## Deployment

This project was deployed on **Dataiku DSS** at Standard Chartered Global Business Services. For standalone deployment:

```bash
# Build Docker container
docker build -t lma-tool:v1 .

# Run locally
docker run -p 8000:8000 lma-tool:v1

# Deploy to Kubernetes (see k8s/ directory)
kubectl apply -f k8s/deployment.yaml
```

---

## Context

Built during a 6-month Data Scientist internship at Standard Chartered Global Business Services, Hyderabad (Jul–Dec 2025). The tool was deployed in production on Dataiku DSS and used daily by the legal compliance team.

**Note:** Training data and fine-tuned model weights are proprietary to Standard Chartered and not included in this repository. The code architecture, preprocessing logic, and methodology are shared here for educational purposes.

---

*Harshal Bhambhani · BITS Hyderabad · 2026*
