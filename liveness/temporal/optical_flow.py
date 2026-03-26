"""
liveness/temporal/optical_flow.py
────────────────────────────────────────────────────────────────────────────
Landmark-based facial deformation liveness detection.

Instead of pixel-level Farneback flow on the whole frame (which picks up
hand tremor and phone movement), we track MediaPipe facial landmark
positions across frames and measure INTERNAL facial deformation only.

KEY INSIGHT
────────────
By subtracting the face centroid before computing displacement, we remove
all rigid body motion (phone moving, hand tremor, camera shake).
Only internal facial deformation remains.

  Printed photo held still:  zero deformation → SPOOF
  Printed photo moved:       centroid subtracted → still zero deformation → SPOOF
  Live face sitting still:   breathing causes ~0.2-0.8px deformation → LIVE
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# Key landmark indices — spread across face for good coverage
# Nose tip, chin, left cheek, right cheek, forehead, eye corners, mouth corners
KEY_LANDMARKS = [1, 152, 234, 454, 10, 33, 263, 61, 291]

MIN_MOTION_MAGNITUDE = 0.0005


class OpticalFlowChecker(BaseLivenessCheck):
    """
    Detects liveness by measuring internal facial landmark deformation.
    Immune to whole-frame motion (phone attacks, hand tremor).
    """

    def __init__(
        self,
        min_frames: int = 8,
        motion_threshold: float = MIN_MOTION_MAGNITUDE,
    ):
        self.min_frames       = min_frames
        self.motion_threshold = motion_threshold
        self._face_mesh       = None

        log.info(
            "OpticalFlowChecker initialised — "
            "motion_threshold=%.5f (landmark mode)",
            motion_threshold,
        )

    @property
    def name(self) -> str:
        return "optical_flow"

    def _load_mediapipe(self) -> None:
        """Lazy-load MediaPipe FaceMesh."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("OpticalFlowChecker: MediaPipe FaceMesh loaded")
        except ImportError:
            raise ImportError("Run: pip install mediapipe==0.10.14")

    def _compute_flow_stats(
        self, frames: list[np.ndarray]
    ) -> dict:
        """
        Compute centroid-normalised landmark displacement across frames.

        Subtracting the centroid removes rigid body motion — only internal
        facial deformation remains, which is zero for a photo.
        """
        self._load_mediapipe()

        landmark_positions = []

        for frame in frames:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                continue

            h, w = frame.shape[:2]
            lm   = result.multi_face_landmarks[0].landmark
            positions = np.array(
                [[lm[i].x * w, lm[i].y * h] for i in KEY_LANDMARKS],
                dtype=np.float32,
            )
            landmark_positions.append(positions)

        if len(landmark_positions) < 2:
            return {
                "mean_magnitude":     0.0,
                "magnitude_variance": 0.0,
                "mean_entropy":       0.0,
                "flow_pairs":         0,
                "magnitudes":         [],
            }

        displacements = []

        for i in range(len(landmark_positions) - 1):
            prev = landmark_positions[i]
            curr = landmark_positions[i + 1]

            # Subtract centroid — removes whole-face translation
            prev_norm = prev - prev.mean(axis=0)
            curr_norm = curr - curr.mean(axis=0)

            # Internal deformation magnitude
            delta     = curr_norm - prev_norm
            magnitude = float(np.linalg.norm(delta, axis=1).mean())
            displacements.append(magnitude)

        displacements = np.array(displacements)

        return {
            "mean_magnitude":     float(displacements.mean()),
            "magnitude_variance": float(displacements.var()),
            "mean_entropy":       float(displacements.std()),
            "flow_pairs":         len(displacements),
            "magnitudes":         displacements.tolist(),
        }

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run landmark deformation liveness check on a frame sequence.

        Args:
            frames: Full BGR camera frames (not aligned crops).
                    Minimum 8 frames recommended.

        Returns:
            Liveness confidence in [0.0, 1.0]:
              0.0 = no facial deformation (static photo)
              0.5+ = facial micro-motion detected (live face)
        """
        if not frames or len(frames) < 2:
            log.warning(
                "OpticalFlowChecker needs ≥2 frames, got %d",
                len(frames) if frames else 0,
            )
            return 0.0

        stats    = self._compute_flow_stats(frames)
        mean_mag = stats["mean_magnitude"]
        variance = stats["magnitude_variance"]

        log.info(
            "Landmark flow — mean_deformation=%.5f, variance=%.6f, pairs=%d",
            mean_mag, variance, stats["flow_pairs"],
        )

        # Gate: no deformation → static image
        if mean_mag < self.motion_threshold:
            log.info(
                "Flow: SPOOF — no facial deformation (%.5f < %.5f)",
                mean_mag, self.motion_threshold,
            )
            return 0.0

        # Score based on deformation magnitude and variance
        magnitude_score = float(min(1.0, mean_mag / 0.5))
        variance_score  = float(min(1.0, variance / 0.01))

        score = float(np.clip(
            0.50 * magnitude_score + 0.50 * variance_score,
            0.0, 1.0,
        ))

        log.info("OpticalFlow score: %.4f", score)
        return score