"""
liveness/temporal/blink_detector.py
────────────────────────────────────────────────────────────────────────────
Blink-based liveness detection using Eye Aspect Ratio (EAR) + MediaPipe.
Updated with face stability gate to prevent false positives from phone
photos being physically moved.

STABILITY GATE
───────────────
A real blink happens while the face is relatively still.
A fake blink from a moving phone happens while the face position
is changing rapidly (hand tremor, repositioning).

We measure std dev of the nose tip landmark position across frames.
If it exceeds stability_threshold pixels, blink counts are invalidated.

  Still face (real blink):   nose_std < 20px → blinks counted ✓
  Moving phone (fake blink): nose_std > 20px → blinks rejected ✗
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# ── MediaPipe landmark indices for eyes ───────────────────────────────────
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

EAR_CLOSED_THRESHOLD = 0.20
MIN_CLOSED_FRAMES    = 1
MAX_CLOSED_FRAMES    = 6
STABILITY_THRESHOLD  = 20.0  # pixels


class BlinkDetector(BaseLivenessCheck):
    """
    Detects natural eye blinks using MediaPipe Face Mesh + EAR.
    Includes stability gate to reject false blinks from moving photos.
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
            "BlinkDetector — ear_threshold=%.2f, stability=%.1fpx",
            ear_threshold, stability_threshold,
        )

    @property
    def name(self) -> str:
        return "blink_ear"

    def _load_mediapipe(self):
        """Lazy-load MediaPipe Face Mesh on first use."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("Run: pip install mediapipe==0.10.14")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        log.info("MediaPipe FaceMesh loaded")

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
        """
        Check if face position is stable enough to trust blink counts.

        Uses nose tip (landmark 1) position variance across frames.
        Returns (is_stable, std_dev_pixels).
        """
        self._load_mediapipe()
        nose_positions = []

        for frame in frames:
            lm = self._get_landmarks_from_frame(frame)
            if lm is not None:
                nose_positions.append(lm[1])  # nose tip

        if len(nose_positions) < 3:
            return True, 0.0  # insufficient data — assume stable

        positions = np.array(nose_positions)
        std_dev   = float(np.std(positions, axis=0).mean())
        is_stable = std_dev < self.stability_threshold

        log.debug(
            "Face stability std=%.2fpx — %s",
            std_dev, "STABLE" if is_stable else "MOVING",
        )
        return is_stable, std_dev

    def count_blinks(
        self,
        frames: list[np.ndarray],
        check_stability: bool = True,
    ) -> tuple[int, list[float]]:
        """
        Count blinks across a frame sequence.

        Args:
            frames:          Full BGR camera frames.
            check_stability: Invalidate blinks if face is moving.

        Returns:
            (blink_count, ear_sequence)
        """
        self._load_mediapipe()

        # Stability gate — reject blinks on moving face
        if check_stability and len(frames) >= 5:
            is_stable, std_dev = self._is_face_stable(frames)
            if not is_stable:
                log.warning(
                    "Face unstable (std=%.1fpx) — blinks invalidated. "
                    "Possible phone/photo attack.",
                    std_dev,
                )
                return 0, []

        ear_sequence:  list[float] = []
        blink_count   = 0
        closed_frames = 0

        for i, frame in enumerate(frames):
            lm = self._get_landmarks_from_frame(frame)
            if lm is None:
                ear_sequence.append(0.0)
                continue

            l_ear   = self._compute_ear(lm, LEFT_EYE_INDICES)
            r_ear   = self._compute_ear(lm, RIGHT_EYE_INDICES)
            avg_ear = (l_ear + r_ear) / 2.0
            ear_sequence.append(avg_ear)

            if avg_ear < self.ear_threshold:
                closed_frames += 1
            else:
                if self.min_closed_frames <= closed_frames <= self.max_closed_frames:
                    blink_count += 1
                    log.debug("Blink #%d at frame %d", blink_count, i)
                closed_frames = 0

        log.info("Blinks=%d over %d frames", blink_count, len(frames))
        return blink_count, ear_sequence

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run blink liveness check with stability gate.

        Returns:
            0.0 = no blinks detected or face was unstable (moving phone)
            0.7-1.0 = confirmed blinks on stable face
        """
        if not frames:
            return 0.0

        blink_count, _ = self.count_blinks(frames, check_stability=True)

        if blink_count == 0:
            score = 0.0
        elif blink_count >= self.min_blinks_required:
            excess = blink_count - self.min_blinks_required
            score  = min(1.0, 0.7 + (excess * 0.15))
        else:
            score = 0.4

        log.info("BlinkDetector score: %.3f (blinks=%d)", score, blink_count)
        return score

    def get_ear_stats(self, frames: list[np.ndarray]) -> dict:
        """Return EAR statistics for debugging/report."""
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