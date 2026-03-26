"""
decision_engine.py
──────────────────
Applies similarity thresholds and policies to produce a final ACCEPT/REJECT verdict.
"""
import numpy as np


class DecisionEngine:
    """Combines liveness and similarity scores into an authentication decision."""

    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold

    def decide(
        self,
        similarity: float,
        liveness_score: float,
        liveness_threshold: float = 0.50,
    ) -> dict:
        """
        Returns:
            {
                "accepted": bool,
                "similarity": float,
                "liveness_score": float,
                "reason": str
            }
        """
        raise NotImplementedError
