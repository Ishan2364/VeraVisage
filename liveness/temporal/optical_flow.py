"""
liveness/temporal/optical_flow.py
────────────────────────────────────────────────────────────────────────────
Skin micro-motion liveness detection using dense optical flow on the
aligned face crop.

THE CORE FIX
─────────────
Old approach: MediaPipe landmark deformation across frames.
Problem:      MediaPipe landmark positions jitter on screen faces,
              creating fake deformation even on static screens.

New approach: Dense Farneback optical flow on the SKIN REGION of the
              aligned 112x112 face crop.

WHY THIS WORKS
──────────────
Real skin has constant micro-motion from:
  - Blood flow causing sub-pixel skin colour changes
  - Breathing causing subtle scale changes
  - Muscle micro-expressions

A screen-displayed or printed face has:
  - Zero skin micro-motion (static pixels)
  - Screen refresh artifacts (high frequency, easily filtered)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# ── Farneback parameters ──────────────────────────────────────────────────
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=2,
    winsize=8,      # small window — sensitive to micro-motion
    iterations=2,
    poly_n=5,
    poly_sigma=1.1,
    flags=0,
)

# Skin ROI on the 112x112 aligned crop
# Central region: nose bridge to chin, avoiding eye/mouth edges
SKIN_ROI = {
    "y1": 40,   # below the eyes
    "y2": 85,   # above the chin
    "x1": 28,   # left edge of nose
    "x2": 84,   # right edge of nose
}

# Thresholds
MIN_SKIN_MOTION     = 0.005    # below this = static (screen/photo)
OPTIMAL_MAX_MOTION  = 0.20     # above this starts getting suspicious (shaky cam)
MAX_SKIN_MOTION     = 3.0      # above this = too much motion (alignment fail)
MIN_MOTION_VARIANCE = 0.00001  # below this = unnaturally uniform


class OpticalFlowChecker(BaseLivenessCheck):
    """
    Detects liveness by measuring skin micro-motion on aligned face crops.

    Works on the 112x112 aligned crop from FaceAligner — not the full frame.
    Immune to whole-frame motion (phone attacks) because we measure
    internal skin texture motion, not positional changes.
    """

    def __init__(
        self,
        min_frames: int = 8,
        motion_threshold: float = MIN_SKIN_MOTION,
    ):
        self.min_frames       = min_frames
        self.motion_threshold = motion_threshold

        log.info(
            "OpticalFlowChecker (skin micro-motion) — "
            "motion_threshold=%.4f",
            motion_threshold,
        )

    @property
    def name(self) -> str:
        return "optical_flow"

    def _extract_skin_roi(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Extract the central skin region, convert to grayscale, and blur 
        slightly to eliminate high-frequency ISO camera noise.
        """
        roi = aligned_crop[
            SKIN_ROI["y1"]:SKIN_ROI["y2"],
            SKIN_ROI["x1"]:SKIN_ROI["x2"],
        ]
        
        # Convert to grayscale for flow computation
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        # Apply slight blur to kill sensor noise, preserving actual physical motion
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        return roi

    def _compute_flow_stats(
        self, aligned_frames: list[np.ndarray]
    ) -> dict:
        """
        Compute dense optical flow statistics on skin ROI across frames.
        """
        empty_stats = {
            "mean_magnitude":     0.0,
            "magnitude_variance": 0.0,
            "flow_pairs":         0,
            "magnitudes":         [],
        }

        if len(aligned_frames) < 2:
            return empty_stats

        # Extract skin ROI from each frame
        roi_frames = []
        for frame in aligned_frames:
            roi = self._extract_skin_roi(frame)
            if roi.size > 0:
                roi_frames.append(roi)

        if len(roi_frames) < 2:
            return empty_stats

        magnitudes = []

        for i in range(len(roi_frames) - 1):
            prev = roi_frames[i]
            curr = roi_frames[i + 1]

            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, **FARNEBACK_PARAMS
            )

            # Separate X and Y motion vectors
            dx = flow[..., 0]
            dy = flow[..., 1]

            # --- THE FIX: GLOBAL MOTION SUBTRACTION ---
            # 1. Find the median movement of the entire ROI (this isolates the hand-shake / box jitter)
            global_dx = np.median(dx)
            global_dy = np.median(dy)

            # 2. Subtract the global movement to isolate purely LOCAL pixel deformation
            local_dx = dx - global_dx
            local_dy = dy - global_dy

            # 3. Calculate magnitude of ONLY the isolated local motion
            local_mag, _ = cv2.cartToPolar(local_dx, local_dy)

            # Use median of the local magnitudes
            median_mag = float(np.median(local_mag))
            magnitudes.append(median_mag)

            log.debug(
                "Skin flow pair %d->%d: median_local_mag=%.5f",
                i, i + 1, median_mag,
            )

        if not magnitudes:
            return empty_stats

        magnitudes_arr = np.array(magnitudes)
        return {
            "mean_magnitude":     float(magnitudes_arr.mean()),
            "magnitude_variance": float(magnitudes_arr.var()),
            "flow_pairs":         len(magnitudes_arr),
            "magnitudes":         magnitudes_arr.tolist(),
        }

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run skin micro-motion liveness check.
        Returns a confidence score in [0.0, 1.0].
        """
        if not frames or len(frames) < 2:
            log.warning(
                "OpticalFlowChecker needs >=2 frames, got %d",
                len(frames) if frames else 0,
            )
            return 0.0

        stats    = self._compute_flow_stats(frames)
        mean_mag = stats["mean_magnitude"]
        variance = stats["magnitude_variance"]

        log.info(
            "Skin flow — mean=%.5f, variance=%.6f, pairs=%d",
            mean_mag, variance, stats["flow_pairs"],
        )

        # Gate 1: No motion → static image (photo or screen)
        if mean_mag < self.motion_threshold:
            log.info(
                "Flow: SPOOF — no skin micro-motion (%.5f < %.4f)",
                mean_mag, self.motion_threshold,
            )
            return 0.0

        # Gate 2: Excessive motion → alignment failure or chaotic movement
        if mean_mag > MAX_SKIN_MOTION:
            log.info("Flow: low confidence — excessive motion")
            return 0.3

        # Gate 3: Uniform motion → possible replay artifact
        if variance < MIN_MOTION_VARIANCE:
            log.info("Flow: suspicious — unnaturally uniform motion")
            return 0.35

        # Goldilocks Scoring: Penalize if motion is unusually high to prevent "shaky camera" bypasses
        if mean_mag <= OPTIMAL_MAX_MOTION:
            # Ramps up normally to 1.0
            magnitude_score = float(min(1.0, mean_mag / 0.1))
        else:
            # Scales linearly down as motion gets chaotic
            magnitude_score = float(max(0.0, 1.0 - ((mean_mag - OPTIMAL_MAX_MOTION) / (MAX_SKIN_MOTION - OPTIMAL_MAX_MOTION))))

        variance_score = float(min(1.0, variance / 0.0005))

        score = float(np.clip(
            0.60 * magnitude_score + 0.40 * variance_score,
            0.0, 1.0,
        ))

        log.info("OpticalFlow (skin) score: %.4f", score)
        return score