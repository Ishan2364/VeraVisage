"""
verification/decision_engine.py
────────────────────────────────────────────────────────────────────────────
Applies threshold policy to produce a final ACCEPT/REJECT verdict.

This module sits at the end of the pipeline. It receives:
  - similarity:      cosine similarity between probe and gallery embedding
  - liveness_passed: bool — did the dual-veto liveness check pass?

And produces a structured verdict dict that the pipeline and API return.

DESIGN PRINCIPLE
─────────────────
The decision engine knows nothing about HOW the similarity was computed
or HOW liveness was checked. It only applies policy. This separation means
you can swap the similarity metric or liveness system without touching
the decision logic.

THRESHOLD TUNING
─────────────────
The default threshold (0.50) is conservative. In production you would
tune this using FAR/FRR curves on a validation set:
  - Lower threshold → more ACCEPTS → higher False Accept Rate (FAR)
  - Higher threshold → more REJECTS → higher False Reject Rate (FRR)

For a security application you want FAR as low as possible.
For a convenience application you want FRR as low as possible.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from infrastructure.logger import get_logger

log = get_logger(__name__)

# Default similarity threshold — tunable via model_config.yaml
DEFAULT_SIMILARITY_THRESHOLD = 0.50


class DecisionEngine:
    """
    Combines liveness and similarity into a final authentication verdict.

    The logic is strictly ordered:
      1. Liveness must have passed (dual-veto cleared)
      2. Similarity must exceed the threshold
      3. Only then: ACCEPT
    """

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """
        Args:
            similarity_threshold: Cosine similarity must exceed this for ACCEPT.
                                  Default 0.50 — adjust based on FAR/FRR testing.
        """
        self.similarity_threshold = similarity_threshold
        log.info(
            "DecisionEngine initialised — similarity_threshold=%.2f",
            similarity_threshold,
        )

    def decide(
        self,
        similarity: float,
        liveness_passed: bool,
        liveness_score: float = 0.0,
        user_id: str = "unknown",
    ) -> dict:
        """
        Produce a structured ACCEPT or REJECT verdict.

        Args:
            similarity:      Cosine similarity from matcher.py [−1, 1].
            liveness_passed: True if dual-veto liveness check cleared.
            liveness_score:  Raw liveness confidence score for logging.
            user_id:         The claimed user identity (for logging).

        Returns:
            Dict with keys:
              "accepted":         bool   — True = ACCEPT, False = REJECT
              "similarity":       float  — cosine similarity score
              "liveness_passed":  bool   — did liveness clear?
              "liveness_score":   float  — raw liveness confidence
              "reason":           str    — human-readable explanation
              "user_id":          str    — the claimed identity
        """

        # Gate 1 — liveness must pass first
        # (dual-veto already happened upstream — this is just a safety check)
        if not liveness_passed:
            result = {
                "accepted":        False,
                "similarity":      round(similarity, 4),
                "liveness_passed": False,
                "liveness_score":  round(liveness_score, 4),
                "reason":          "Liveness check failed — not a live person.",
                "user_id":         user_id,
            }
            log.warning(
                "REJECT '%s' — liveness failed (score=%.3f)",
                user_id, liveness_score,
            )
            return result

        # Gate 2 — identity similarity threshold
        if similarity < self.similarity_threshold:
            result = {
                "accepted":        False,
                "similarity":      round(similarity, 4),
                "liveness_passed": True,
                "liveness_score":  round(liveness_score, 4),
                "reason": (
                    f"Identity mismatch — similarity {similarity:.3f} "
                    f"< threshold {self.similarity_threshold:.2f}. "
                    f"Live person confirmed but not the enrolled user."
                ),
                "user_id": user_id,
            }
            log.warning(
                "REJECT '%s' — similarity too low (%.3f < %.2f)",
                user_id, similarity, self.similarity_threshold,
            )
            return result

        # Both gates passed → ACCEPT
        result = {
            "accepted":        True,
            "similarity":      round(similarity, 4),
            "liveness_passed": True,
            "liveness_score":  round(liveness_score, 4),
            "reason": (
                f"Identity verified — similarity {similarity:.3f} "
                f"≥ threshold {self.similarity_threshold:.2f}. "
                f"Liveness confirmed."
            ),
            "user_id": user_id,
        }
        log.info(
            "ACCEPT '%s' — similarity=%.3f, liveness_score=%.3f",
            user_id, similarity, liveness_score,
        )
        return result