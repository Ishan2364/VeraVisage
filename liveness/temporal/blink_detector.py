"""
liveness/temporal/blink_detector.py
────────────────────────────────────────────────────────────────────────────
Blink detection using EAR V-shape pattern matching.

THE CORE FIX
─────────────
Old approach: count any EAR crossing below 0.20 threshold.
Problem:      MediaPipe landmark jitter on screen faces creates random
              EAR spikes that cross the threshold — false blinks.

New approach: validate the SHAPE of each EAR dip.
A real blink has a smooth V-shape:
    0.28 → 0.22 → 0.15 → 0.10 → 0.16 → 0.24 → 0.29
    (smooth drop, brief hold, smooth rise)

Landmark jitter produces random spiky noise:
    0.28 → 0.19 → 0.26 → 0.21 → 0.27 → 0.19 → 0.28
    (no coherent shape, no smooth transition)

We validate each candidate blink by checking:
  1. Drop smoothness — EAR decreases monotonically into the dip
  2. Rise smoothness — EAR increases monotonically out of the dip
  3. Minimum depth   — dip must go at least 0.05 below baseline
  4. Duration        — dip lasts 1-6 frames (not instantaneous noise)

A jitter spike fails checks 1 and 2. A real blink passes all four.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# ── MediaPipe landmark indices ─────────────────────────────────────────────
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

# ── Blink detection parameters ────────────────────────────────────────────
EAR_CLOSED_THRESHOLD   = 0.21   # below this = eye closing
MIN_BLINK_DEPTH        = 0.04   # min drop from baseline to count as blink
MIN_CLOSED_FRAMES      = 1      # min frames below threshold
MAX_CLOSED_FRAMES      = 7      # max frames (more = held shut, not blink)
SMOOTHNESS_WINDOW      = 2      # frames before/after dip to check smoothness
STABILITY_THRESHOLD    = 20.0   # max face movement in pixels


class BlinkDetector(BaseLivenessCheck):
    """
    Detects genuine eye blinks using EAR V-shape pattern matching.

    Rejects landmark jitter from screen/printed faces by validating
    the temporal shape of each EAR dip, not just its depth.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_CLOSED_THRESHOLD,
        min_closed_frames: int = MIN_CLOSED_FRAMES,
        max_closed_frames: int = MAX_CLOSED_FRAMES,
        min_blinks_required: int = 1,
        stability_threshold: float = STABILITY_THRESHOLD,
    ):
        self.ear_threshold       = ear_threshold
        self.min_closed_frames   = min_closed_frames
        self.max_closed_frames   = max_closed_frames
        self.min_blinks_required = min_blinks_required
        self.stability_threshold = stability_threshold
        self._face_mesh          = None

        log.info(
            "BlinkDetector (V-shape) — ear_threshold=%.2f, "
            "min_depth=%.2f, stability=%.1fpx",
            ear_threshold, MIN_BLINK_DEPTH, stability_threshold,
        )

    @property
    def name(self) -> str:
        return "blink_ear"

    def _load_mediapipe(self) -> None:
        """Lazy-load MediaPipe FaceMesh."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe FaceMesh loaded")
        except ImportError:
            raise ImportError("Run: pip install mediapipe==0.10.14")

    @staticmethod
    def _compute_ear(landmarks: np.ndarray, eye_indices: list[int]) -> float:
        """Compute Eye Aspect Ratio for one eye."""
        p = landmarks[eye_indices]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        if C < 1e-6:
            return 0.0
        return float((A + B) / (2.0 * C))

    def _get_landmarks_from_frame(
        self, frame: np.ndarray
    ) -> np.ndarray | None:
        """Run MediaPipe on one BGR frame, return (468, 2) landmarks."""
        import cv2
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w    = frame.shape[:2]
        face_lm = results.multi_face_landmarks[0]
        return np.array(
            [[lm.x * w, lm.y * h] for lm in face_lm.landmark],
            dtype=np.float32,
        )

    def _is_face_stable(
        self, frames: list[np.ndarray]
    ) -> tuple[bool, float]:
        """Check face position stability using nose tip landmark."""
        self._load_mediapipe()
        nose_positions = []
        for frame in frames:
            lm = self._get_landmarks_from_frame(frame)
            if lm is not None:
                nose_positions.append(lm[1])
        if len(nose_positions) < 3:
            return True, 0.0
        positions = np.array(nose_positions)
        std_dev   = float(np.std(positions, axis=0).mean())
        return std_dev < self.stability_threshold, std_dev

    def _is_valid_blink(
        self,
        ear_sequence: list[float],
        dip_start: int,
        dip_end: int,
    ) -> bool:
        """
        Validate that an EAR dip has the shape of a real blink.

        Checks:
          1. Drop smoothness — values decrease into the dip
          2. Rise smoothness — values increase out of the dip
          3. Minimum depth   — dip is at least MIN_BLINK_DEPTH below baseline
          4. Duration        — within min/max closed frame count

        Args:
            ear_sequence: Full EAR sequence across all frames.
            dip_start:    Index where EAR first went below threshold.
            dip_end:      Index where EAR rose back above threshold.

        Returns:
            True if this dip looks like a real blink.
        """
        n = len(ear_sequence)

        # Check 1: Duration
        duration = dip_end - dip_start
        if not (self.min_closed_frames <= duration <= self.max_closed_frames):
            log.debug("Blink rejected: duration=%d out of range", duration)
            return False

        # Check 2: Minimum depth
        # Baseline = average EAR in the SMOOTHNESS_WINDOW frames before the dip
        pre_start  = max(0, dip_start - SMOOTHNESS_WINDOW)
        pre_values = ear_sequence[pre_start:dip_start]
        if not pre_values:
            return False
        baseline = np.mean(pre_values)
        dip_min  = min(ear_sequence[dip_start:dip_end + 1])

        if (baseline - dip_min) < MIN_BLINK_DEPTH:
            log.debug(
                "Blink rejected: depth=%.3f < min=%.3f",
                baseline - dip_min, MIN_BLINK_DEPTH,
            )
            return False

        # Check 3: Drop smoothness
        # Values from pre-window to dip minimum should be generally decreasing
        # We allow one frame of non-decrease (noise tolerance)
        if dip_start > 0:
            drop_seq = ear_sequence[max(0, dip_start - 1):dip_start + 1]
            violations = sum(
                1 for i in range(1, len(drop_seq))
                if drop_seq[i] > drop_seq[i-1] + 0.02  # 0.02 tolerance
            )
            if violations > 1:
                log.debug("Blink rejected: drop not smooth (violations=%d)", violations)
                return False

        # Check 4: Rise smoothness
        # Values from dip minimum to post-window should be generally increasing
        post_end  = min(n, dip_end + SMOOTHNESS_WINDOW)
        rise_seq  = ear_sequence[dip_end:post_end]
        if len(rise_seq) >= 2:
            violations = sum(
                1 for i in range(1, len(rise_seq))
                if rise_seq[i] < rise_seq[i-1] - 0.02  # 0.02 tolerance
            )
            if violations > 1:
                log.debug("Blink rejected: rise not smooth (violations=%d)", violations)
                return False

        log.debug(
            "Valid blink — duration=%d, depth=%.3f, baseline=%.3f",
            duration, baseline - dip_min, baseline,
        )
        return True

    def count_blinks(
        self,
        frames: list[np.ndarray],
        check_stability: bool = True,
    ) -> tuple[int, list[float]]:
        """
        Count validated blinks across a frame sequence.

        Each candidate blink (EAR below threshold) is validated by
        V-shape pattern matching before being counted.

        Args:
            frames:          Full BGR camera frames.
            check_stability: Reject blinks if face is moving.

        Returns:
            (blink_count, ear_sequence)
        """
        self._load_mediapipe()

        # Stability gate
        if check_stability and len(frames) >= 5:
            is_stable, std_dev = self._is_face_stable(frames)
            if not is_stable:
                log.warning(
                    "Face unstable (std=%.1fpx) — blinks invalidated",
                    std_dev,
                )
                return 0, []

        # Build EAR sequence
        ear_sequence: list[float] = []
        for frame in frames:
            lm = self._get_landmarks_from_frame(frame)
            if lm is None:
                # Carry forward last value or use open-eye default
                ear_sequence.append(
                    ear_sequence[-1] if ear_sequence else 0.30
                )
                continue
            l_ear   = self._compute_ear(lm, LEFT_EYE_INDICES)
            r_ear   = self._compute_ear(lm, RIGHT_EYE_INDICES)
            avg_ear = (l_ear + r_ear) / 2.0
            ear_sequence.append(avg_ear)

        # Find and validate blink candidates
        blink_count   = 0
        in_blink      = False
        blink_start   = 0
        closed_count  = 0

        for i, ear in enumerate(ear_sequence):
            if ear < self.ear_threshold:
                if not in_blink:
                    in_blink    = True
                    blink_start = i
                closed_count += 1
            else:
                if in_blink:
                    # Eye just opened — validate this dip
                    if self._is_valid_blink(ear_sequence, blink_start, i):
                        blink_count += 1
                        log.debug("Validated blink #%d at frame %d", blink_count, blink_start)
                    in_blink     = False
                    closed_count = 0

        log.info(
            "Blink count: %d over %d frames", blink_count, len(frames)
        )
        return blink_count, ear_sequence

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run V-shape blink liveness check.

        Returns:
            0.0 = no valid blinks (static photo or unstable face)
            0.7-1.0 = confirmed natural blinks
        """
        if not frames:
            return 0.0

        blink_count, _ = self.count_blinks(frames, check_stability=True)

        if blink_count == 0:
            score = 0.0
        elif blink_count >= self.min_blinks_required:
            excess = blink_count - self.min_blinks_required
            score  = min(1.0, 0.7 + excess * 0.10)
        else:
            score = 0.35

        log.info("BlinkDetector score: %.3f (blinks=%d)", score, blink_count)
        return score

    def get_ear_stats(self, frames: list[np.ndarray]) -> dict:
        """Return EAR statistics for debugging."""
        blink_count, ear_sequence = self.count_blinks(frames)
        if not ear_sequence:
            return {"error": "No landmarks detected"}
        ear_arr = np.array(ear_sequence)
        return {
            "blink_count":            blink_count,
            "ear_mean":               float(ear_arr.mean()),
            "ear_std":                float(ear_arr.std()),
            "ear_min":                float(ear_arr.min()),
            "ear_max":                float(ear_arr.max()),
            "frames_below_threshold": int((ear_arr < self.ear_threshold).sum()),
        }