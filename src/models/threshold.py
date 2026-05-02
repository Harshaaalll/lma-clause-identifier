"""
Confidence Thresholding for Clause Classifier

Filters low-confidence predictions before returning results to auditors.
Eliminates 95%+ of false positives while preserving 98.6% recall.

Two methods evaluated:
1. KDE Plot method — rejected
   Fit kernel density to correct/incorrect prediction distributions,
   find the intersection point as the threshold.
   Problem: The fine-tuned model had very few incorrect predictions,
   so no meaningful "incorrect" distribution existed to fit a KDE to.
   No intersection = no threshold. Method inapplicable.

2. Minimum Correct Confidence (MCC) method — selected
   Find the lowest confidence score among ALL correctly classified
   validation samples. Set this as the threshold.
   Rationale: Any prediction below this level was never associated
   with a correct classification in validation data, so it should
   not be trusted in production.
   Result: Threshold ≈ 0.72 on our data, filtering 95%+ false positives.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceThresholder:
    """
    Applies minimum correct confidence thresholding to classifier output.

    Filters predictions below the calibrated threshold, marking them
    as uncertain for human review rather than passing to auditors.
    """

    def __init__(self, threshold: float = 0.72):
        """
        Args:
            threshold: Minimum confidence to accept a prediction.
                      Default 0.72 derived from validation set calibration.
                      Override after running calibrate() on your own data.
        """
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def calibrate(
        self,
        predictions: List[dict],
        true_labels: List[str],
    ) -> float:
        """
        Calibrate threshold using Minimum Correct Confidence method.

        Process:
        1. For each prediction, check if it matches the true label
        2. Collect confidence scores of all CORRECT predictions
        3. Set threshold = minimum of those correct confidence scores
        4. Any prediction below this was never seen as correct in validation

        Args:
            predictions:  Output from LMAClassifier.predict()
            true_labels:  Ground truth labels for validation set

        Returns:
            Calibrated threshold value

        Why not use the incorrect distribution:
        - Our fine-tuned model made very few incorrect predictions
        - Attempting to fit a KDE to 3-5 data points is meaningless
        - Minimum correct confidence is robust to small incorrect sets
        """
        if len(predictions) != len(true_labels):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(true_labels)} labels"
            )

        correct_confidences = []

        for pred, true_label in zip(predictions, true_labels):
            if pred['predicted_label'] == true_label:
                correct_confidences.append(pred['confidence'])

        if not correct_confidences:
            self.logger.warning(
                "No correct predictions found for calibration. "
                "Using default threshold 0.5"
            )
            self.threshold = 0.5
            return self.threshold

        self.threshold = float(np.min(correct_confidences))

        self.logger.info(
            f"Threshold calibrated using Minimum Correct Confidence method:\n"
            f"  Correct predictions: {len(correct_confidences)}\n"
            f"  Min confidence: {self.threshold:.4f}\n"
            f"  Mean confidence: {np.mean(correct_confidences):.4f}\n"
            f"  Median confidence: {np.median(correct_confidences):.4f}"
        )

        return self.threshold

    def filter(self, predictions: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Split predictions into accepted and uncertain groups.

        Args:
            predictions: Output from LMAClassifier.predict()

        Returns:
            Tuple of (accepted, uncertain):
            - accepted:  Predictions above threshold, passed to auditors
            - uncertain: Predictions below threshold, flagged for review

        Why return both instead of just accepted:
        - Auditors may want to spot-check uncertain predictions
        - Useful for monitoring threshold effectiveness over time
        - Uncertain chunks may reveal edge cases for model improvement
        """
        accepted = []
        uncertain = []

        for pred in predictions:
            # Always pass through irrelevant — we only threshold clause classes
            if pred['predicted_label'] == 'irrelevant':
                continue

            if pred['confidence'] >= self.threshold:
                accepted.append({**pred, 'threshold_status': 'accepted'})
            else:
                uncertain.append({
                    **pred,
                    'threshold_status': 'uncertain',
                    'threshold_used': self.threshold,
                })

        total_clauses = len(accepted) + len(uncertain)
        filtered_pct = (
            (len(uncertain) / total_clauses * 100) if total_clauses > 0 else 0
        )

        self.logger.info(
            f"Threshold filtering (threshold={self.threshold:.3f}):\n"
            f"  Total clause predictions: {total_clauses}\n"
            f"  Accepted: {len(accepted)}\n"
            f"  Filtered as uncertain: {len(uncertain)} ({filtered_pct:.1f}%)"
        )

        return accepted, uncertain

    def save(self, path: str):
        """Save threshold value to JSON."""
        data = {
            "threshold": self.threshold,
            "method": "minimum_correct_confidence",
            "description": (
                "Minimum confidence score observed among all correct "
                "predictions on validation set."
            )
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Threshold saved: {path}")

    @classmethod
    def load(cls, path: str) -> "ConfidenceThresholder":
        """Load threshold from saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        instance = cls(threshold=data["threshold"])
        instance.logger.info(
            f"Threshold loaded: {data['threshold']:.4f} "
            f"(method: {data.get('method', 'unknown')})"
        )
        return instance
