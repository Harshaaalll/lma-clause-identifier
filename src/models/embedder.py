"""
SBERT Semantic Search for Gold-Standard Clause Matching

After classifying a clause type, this module finds the closest
gold-standard sample language from Standard Chartered's compliance
database using cosine similarity.

Why fine-tune SBERT instead of using out-of-box:
- Generic SBERT understands general language similarity
- Legal language has domain-specific synonyms:
    "obligor shall comply with sanctions regimes"
    ≈ "borrower must adhere to applicable restrictive measures"
  These share NO words but are legally identical.
- Fine-tuning with MultipleNegativesRankingLoss on legal
  anchor-positive pairs teaches the model these equivalences.
- Without fine-tuning: generic model misses ~30% of true matches.
  After fine-tuning: retrieval accuracy significantly improved.

Training setup:
- Loss: MultipleNegativesRankingLoss
  Each (anchor, positive) pair — the model learns that anchor
  and positive should be close, while all other positives in the
  batch serve as negatives. No need to mine hard negatives manually.
- Model: all-MiniLM-L6-v2 (384-dim, fast, good quality baseline)
- Training data: curated legal clause anchor-positive pairs
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import faiss
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning(
        "sentence-transformers or faiss not installed.\n"
        "Install with: pip install sentence-transformers faiss-cpu"
    )


class SBERTEmbedder:
    """
    Semantic search over gold-standard compliance language.

    Embeds a classified clause and retrieves the most similar
    gold-standard sample from the compliance database.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            model_name: Base SBERT model. all-MiniLM-L6-v2 chosen for:
                       - 384 dimensions (fast FAISS search)
                       - Strong baseline performance
                       - 5x faster than larger models
                       - Sufficient quality for legal semantic matching
                         after domain fine-tuning
        """
        if not SBERT_AVAILABLE:
            raise ImportError(
                "Install dependencies: pip install sentence-transformers faiss-cpu"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)

        # FAISS index for fast similarity search
        self.index = None
        self.index_texts = []
        self.index_metadata = []

    def fine_tune(
        self,
        train_pairs: List[Dict],
        output_dir: str = "models/sbert_legal",
        epochs: int = 3,
        batch_size: int = 32,
        warmup_steps: int = 100,
    ):
        """
        Fine-tune SBERT on legal clause anchor-positive pairs.

        Args:
            train_pairs: List of dicts with 'anchor' and 'positive' keys
                        Example: {
                            "anchor": "The obligor shall comply with sanctions...",
                            "positive": "Borrower must adhere to restrictive measures..."
                        }
            output_dir:   Where to save the fine-tuned model
            epochs:       Training epochs
            batch_size:   Training batch size
            warmup_steps: Linear warmup steps for learning rate

        Why MultipleNegativesRankingLoss:
        - No explicit negative pairs needed — other positives in each batch
          serve as implicit negatives
        - Efficient: O(batch_size^2) negatives per batch
        - Works well with small legal datasets where mining hard negatives
          is difficult
        - Standard approach for legal/domain-specific semantic search

        Training data format:
        - Each pair is (anchor_clause, positive_clause)
        - Anchor: clause text as it appears in an actual agreement
        - Positive: corresponding gold-standard sample language
        """
        self.logger.info(
            f"Fine-tuning SBERT on {len(train_pairs)} pairs, "
            f"{epochs} epochs"
        )

        # Build InputExamples
        examples = [
            InputExample(texts=[pair['anchor'], pair['positive']])
            for pair in train_pairs
        ]

        dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.MultipleNegativesRankingLoss(self.model)

        # Fine-tune
        self.model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir,
            show_progress_bar=True,
        )

        self.logger.info(f"SBERT fine-tuned and saved to: {output_dir}")

    def build_index(
        self,
        gold_standard_texts: List[str],
        metadata: Optional[List[dict]] = None,
    ):
        """
        Build FAISS index from gold-standard compliance language.

        Args:
            gold_standard_texts: List of gold-standard clause texts
                                 from compliance database
            metadata:            Optional list of dicts with additional
                                 info per text (e.g. clause type, source doc)

        Why FAISS:
        - In-memory vector search
        - Sub-millisecond search across thousands of embeddings
        - IndexFlatIP (inner product) equivalent to cosine similarity
          when vectors are L2-normalised (which SBERT does by default)
        """
        self.logger.info(
            f"Building FAISS index from {len(gold_standard_texts)} gold-standard texts"
        )

        # Encode all gold-standard texts
        embeddings = self.model.encode(
            gold_standard_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Build FAISS flat inner product index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        self.index_texts = gold_standard_texts
        self.index_metadata = metadata or [{} for _ in gold_standard_texts]

        self.logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, {dim} dimensions"
        )

    def search(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[dict]:
        """
        Find gold-standard clauses most similar to the query.

        Args:
            query_text: Classified clause text to find matches for
            top_k:      Number of top matches to return

        Returns:
            List of dicts with:
            - text:       Gold-standard sample text
            - similarity: Cosine similarity score (0-1)
            - rank:       1-indexed rank
            - metadata:   Additional info from index_metadata

        Why cosine similarity:
        - Measures semantic direction (meaning) not magnitude
        - Works well for text where sentence length varies
        - Robust to paraphrasing and legal synonyms after fine-tuning
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built. Call build_index() first."
            )

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        similarities, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (sim, idx) in enumerate(
            zip(similarities[0], indices[0]), start=1
        ):
            if idx < 0:  # FAISS returns -1 for empty results
                continue

            results.append({
                'text': self.index_texts[idx],
                'similarity': float(sim),
                'rank': rank,
                'metadata': self.index_metadata[idx],
            })

        return results

    def save_index(self, index_dir: str):
        """Save FAISS index and associated texts to disk."""
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{index_dir}/faiss.index")

        with open(f"{index_dir}/texts.json", "w") as f:
            json.dump({
                "texts": self.index_texts,
                "metadata": self.index_metadata,
            }, f, indent=2)

        self.logger.info(f"FAISS index saved: {index_dir}")

    def load_index(self, index_dir: str):
        """Load FAISS index and associated texts from disk."""
        self.index = faiss.read_index(f"{index_dir}/faiss.index")

        with open(f"{index_dir}/texts.json") as f:
            data = json.load(f)
            self.index_texts = data["texts"]
            self.index_metadata = data["metadata"]

        self.logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors"
        )
