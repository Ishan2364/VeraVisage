"""
liveness/liveness_aggregator.py
────────────────────────────────────────────────────────────────────────────
Fuses scores from all registered liveness checks into a single verdict.

HOW FUSION WORKS
─────────────────
Each check returns a score in [0, 1]. The aggregator computes a weighted
average. The weights are loaded from model_config.yaml so you can tune
them without touching code.

Example with two checks:
    texture score = 0.85, weight = 0.50
    blink score   = 1.00, weight = 0.50
    ──────────────────────────────────
    fused score   = (0.85 × 0.50) + (1.00 × 0.50) = 0.925

If fused score >= liveness_threshold → LIVE
If fused score <  liveness_threshold → SPOOF

EARLY REJECTION
────────────────
Any check that scores below its individual hard_reject_threshold causes
an immediate SPOOF verdict regardless of other check scores. This prevents
a perfect blink score from masking a terrible texture score — if texture
says 0.05 (almost certainly a photo), we don't need to average it.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# Any single check scoring below this triggers immediate rejection
HARD_REJECT_THRESHOLD = 0.15


class LivenessAggregator:
    """
    Runs all registered liveness checks and returns a fused verdict.

    Usage:
        aggregator = LivenessAggregator(
            checks=[texture_analyzer, blink_detector],
            weights={"texture_lbp": 0.50, "blink_ear": 0.50},
            liveness_threshold=0.60,
        )
        is_live, score, breakdown = aggregator.evaluate(frames, aligned_frames)
    """

    def __init__(
        self,
        checks: list[BaseLivenessCheck],
        weights: dict[str, float],
        liveness_threshold: float = 0.60,
        hard_reject_threshold: float = HARD_REJECT_THRESHOLD,
    ):
        """
        Args:
            checks:                List of liveness check objects.
            weights:               Dict mapping check.name → weight float.
                                   Should sum to 1.0 (normalised internally
                                   if they don't).
            liveness_threshold:    Fused score must exceed this to pass.
            hard_reject_threshold: Any single check below this = instant fail.
        """
        self.checks = checks
        self.weights = weights
        self.liveness_threshold = liveness_threshold
        self.hard_reject_threshold = hard_reject_threshold

        # Validate weights
        missing = [c.name for c in checks if c.name not in weights]
        if missing:
            log.warning(
                "No weight defined for checks: %s — defaulting to equal weights",
                missing,
            )
            for name in missing:
                weights[name] = 1.0 / len(checks)

        # Normalise weights so they sum to 1.0
        total = sum(weights[c.name] for c in checks)
        if abs(total - 1.0) > 0.01:
            log.warning(
                "Weights sum to %.3f (not 1.0) — normalising automatically",
                total,
            )
            self.weights = {
                name: w / total for name, w in weights.items()
            }

        log.info(
            "LivenessAggregator initialised — checks=%s, threshold=%.2f",
            [c.name for c in checks], liveness_threshold,
        )

    @timer
    def evaluate(
        self,
        frames: list[np.ndarray],
        aligned_frames: list[np.ndarray] | None = None,
    ) -> tuple[bool, float, dict]:
        """
        Run all checks and return a fused liveness verdict.

        Args:
            frames:         Full BGR camera frames (for temporal checks like
                            blink detection and optical flow).
            aligned_frames: 112×112 aligned face crops (for spatial checks
                            like texture analysis and FFT).
                            If None, uses frames for all checks.

        Returns:
            Tuple of:
              is_live   (bool)  — True if the person passed liveness
              score     (float) — weighted fused confidence in [0, 1]
              breakdown (dict)  — per-check scores for logging/explainability
                                  Example:
                                  {
                                    "texture_lbp": 0.85,
                                    "blink_ear":   1.00,
                                    "fused":       0.925,
                                    "passed":      True,
                                    "hard_rejected_by": None,
                                  }
        """
        if aligned_frames is None:
            aligned_frames = frames

        breakdown: dict = {}
        hard_rejected_by: str | None = None

        # ── Run each check ────────────────────────────────────────────────
        for check in self.checks:
            try:
                # Spatial checks (texture, FFT) work on aligned face crops
                # Temporal checks (blink, flow) work on full frames
                # We pass aligned_frames to spatial checks by convention:
                # spatial checks only look at the first few frames anyway
                if check.name in ("texture_lbp", "fft_artifact", "depth_estimation"):
                    score = check.check(aligned_frames)
                else:
                    score = check.check(frames)

                breakdown[check.name] = round(float(score), 4)
                log.info("Check '%s' score: %.4f", check.name, score)

                # Hard rejection: if any single check is catastrophically bad
                if score < self.hard_reject_threshold:
                    hard_rejected_by = check.name
                    log.warning(
                        "Hard rejection triggered by '%s' (score=%.4f < %.4f)",
                        check.name, score, self.hard_reject_threshold,
                    )
                    break  # no point running other checks

            except Exception as e:
                log.error(
                    "Check '%s' raised an exception: %s — scoring as 0.0",
                    check.name, e,
                )
                breakdown[check.name] = 0.0

        # ── Fuse scores ───────────────────────────────────────────────────
        if hard_rejected_by:
            fused_score = 0.0
            is_live = False
        else:
            weighted_sum = sum(
                breakdown.get(c.name, 0.0) * self.weights.get(c.name, 0.0)
                for c in self.checks
            )
            fused_score = float(np.clip(weighted_sum, 0.0, 1.0))
            is_live = fused_score >= self.liveness_threshold

        breakdown["fused"]            = round(fused_score, 4)
        breakdown["passed"]           = is_live
        breakdown["hard_rejected_by"] = hard_rejected_by

        log.info(
            "Liveness result: %s (fused_score=%.4f, threshold=%.2f)",
            "LIVE" if is_live else "SPOOF",
            fused_score,
            self.liveness_threshold,
        )

        return is_live, fused_score, breakdown