"""
liveness/temporal/rppg_detector.py
────────────────────────────────────────────────────────────────────────────
Remote Photoplethysmography (rPPG) liveness detection.

HOW IT WORKS
────────────
Blood flowing through facial skin capillaries causes subtle periodic
changes in skin colour — particularly in the green channel, which is most
sensitive to haemoglobin absorption. A live face has a detectable pulse
signal (typically 60–120 BPM). A printed photo or screen replay does not.

ALGORITHM (Green Channel rPPG)
───────────────────────────────
1. For each frame, extract the mean green channel value from the forehead
   region of interest (ROI) — forehead is chosen because it has the least
   facial hair and most consistent skin exposure
2. Collect these values across N frames to form a raw signal
3. Detrend the signal (remove slow drift from lighting changes)
4. Apply a bandpass filter (0.7–3.5 Hz = 42–210 BPM) to isolate pulse
5. Compute the power spectral density of the filtered signal
6. Check if there is a dominant frequency peak in the valid pulse range
7. Score = how strong and clear that peak is

WHAT IT CATCHES
────────────────
- Printed photos:     No blood flow → no periodic signal → low score
- Static screens:     Uniform colour → no signal → low score
- Video replays:      Screen refresh creates 50/60Hz artifact, not pulse
- Deepfakes:          No real blood flow, no pulse in skin pixels

REQUIREMENTS
─────────────
- Minimum ~10 seconds of video at 25+ FPS for reliable detection
- Good lighting — dim rooms reduce signal quality significantly
- Face must be relatively still — motion artifacts corrupt the signal
- Works best on forehead region (less hair, less expression movement)

REFERENCES
──────────
- Verkruysse et al. (2008) "Remote plethysmographic imaging using ambient light"
- De Haan & Jeanne (2013) "Robust pulse rate from chrominance-based rPPG"
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

import cv2

from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# ── Signal processing constants ───────────────────────────────────────────
# Valid human heart rate range in Hz
PULSE_FREQ_LOW  = 0.7   # 42 BPM
PULSE_FREQ_HIGH = 3.5   # 210 BPM

# Minimum frames needed for reliable frequency analysis
MIN_FRAMES_FOR_RPPG = 60   # ~2 seconds at 30fps

# SNR threshold — signal to noise ratio of the pulse peak
# Below this = noise dominates, no clear pulse detected
MIN_SNR_THRESHOLD = 1.5

# Forehead ROI as fraction of face bounding box
# (top 35% of face height, middle 60% of face width)
FOREHEAD_TOP    = 0.10
FOREHEAD_BOTTOM = 0.35
FOREHEAD_LEFT   = 0.20
FOREHEAD_RIGHT  = 0.80


class RPPGDetector(BaseLivenessCheck):
    """
    Detects liveness by measuring remote photoplethysmography pulse signal.

    No extra hardware required — works with a standard RGB webcam.
    Requires ~60+ frames (~2 seconds) for reliable detection.
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_frames: int = MIN_FRAMES_FOR_RPPG,
        min_snr: float = MIN_SNR_THRESHOLD,
    ):
        """
        Args:
            fps:        Camera frames per second. Used for frequency analysis.
                        30.0 is the standard webcam rate.
            min_frames: Minimum frames required. Returns 0.5 (neutral) if
                        fewer frames are provided.
            min_snr:    Minimum signal-to-noise ratio for a valid pulse peak.
        """
        self.fps        = fps
        self.min_frames = min_frames
        self.min_snr    = min_snr

        log.info(
            "RPPGDetector initialised — fps=%.1f, min_frames=%d, min_snr=%.2f",
            fps, min_frames, min_snr,
        )

    @property
    def name(self) -> str:
        return "rppg_pulse"

    def _extract_forehead_roi(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        """
        Extract the forehead region from a frame.

        Args:
            frame: Full BGR camera frame.
            bbox:  Optional (x1, y1, x2, y2) face bounding box.
                   If None, uses the centre-top region of the frame.

        Returns:
            Forehead ROI as BGR array, or None if extraction fails.
        """
        h, w = frame.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_h = y2 - y1
            face_w = x2 - x1

            # Forehead = top portion of face bounding box
            roi_y1 = int(y1 + face_h * FOREHEAD_TOP)
            roi_y2 = int(y1 + face_h * FOREHEAD_BOTTOM)
            roi_x1 = int(x1 + face_w * FOREHEAD_LEFT)
            roi_x2 = int(x1 + face_w * FOREHEAD_RIGHT)
        else:
            # Fallback: assume face is centred, use top-centre region
            roi_y1 = int(h * 0.15)
            roi_y2 = int(h * 0.35)
            roi_x1 = int(w * 0.30)
            roi_x2 = int(w * 0.70)

        # Bounds check
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(h, roi_y2)
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(w, roi_x2)

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return None

        return frame[roi_y1:roi_y2, roi_x1:roi_x2]

    def _extract_green_signal(
        self,
        frames: list[np.ndarray],
        bboxes: list[tuple] | None = None,
    ) -> np.ndarray:
        """
        Extract mean green channel value from forehead ROI for each frame.

        Args:
            frames: List of BGR frames.
            bboxes: Optional list of (x1,y1,x2,y2) per frame.

        Returns:
            1-D array of mean green values, one per frame.
        """
        green_values = []

        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            roi  = self._extract_forehead_roi(frame, bbox)

            if roi is None or roi.size == 0:
                # Use last valid value or 0
                green_values.append(green_values[-1] if green_values else 0.0)
                continue

            # Mean of green channel (index 1 in BGR)
            mean_green = float(np.mean(roi[:, :, 1]))
            green_values.append(mean_green)

        return np.array(green_values, dtype=np.float64)

    def _bandpass_filter(
        self, signal_raw: np.ndarray, fps: float
    ) -> np.ndarray:
        """
        Apply a bandpass filter to isolate the pulse frequency range.

        Uses a 4th-order Butterworth filter in the range
        [PULSE_FREQ_LOW, PULSE_FREQ_HIGH] Hz.
        """
        nyquist  = fps / 2.0
        low      = PULSE_FREQ_LOW  / nyquist
        high     = PULSE_FREQ_HIGH / nyquist

        # Clamp to valid range (0, 1) exclusive
        low  = np.clip(low,  0.01, 0.99)
        high = np.clip(high, 0.01, 0.99)

        if low >= high:
            return signal_raw

        b, a = scipy_signal.butter(4, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, signal_raw)

    def _compute_pulse_snr(
        self, filtered_signal: np.ndarray, fps: float
    ) -> tuple[float, float]:
        """
        Compute the signal-to-noise ratio of the dominant pulse frequency.

        Returns:
            Tuple of (snr, dominant_bpm):
              snr:          Power of pulse peak / mean noise power.
                            > 1.5 = likely real pulse present.
              dominant_bpm: Estimated heart rate in BPM.
        """
        n = len(filtered_signal)
        if n < 4:
            return 0.0, 0.0

        # Power spectral density via FFT
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        psd   = np.abs(np.fft.rfft(filtered_signal)) ** 2

        # Mask to valid pulse frequency range
        pulse_mask = (freqs >= PULSE_FREQ_LOW) & (freqs <= PULSE_FREQ_HIGH)

        if not pulse_mask.any():
            return 0.0, 0.0

        pulse_psd = psd[pulse_mask]
        pulse_freqs = freqs[pulse_mask]

        # Peak power in pulse band
        peak_idx   = np.argmax(pulse_psd)
        peak_power = pulse_psd[peak_idx]
        peak_freq  = pulse_freqs[peak_idx]

        # Noise = mean power outside the peak ± 0.1 Hz window
        noise_mask = pulse_mask.copy()
        # Exclude the peak and its neighbours
        peak_global_idx = np.where(pulse_mask)[0][peak_idx]
        for offset in range(-3, 4):
            idx = peak_global_idx + offset
            if 0 <= idx < len(noise_mask):
                noise_mask[idx] = False

        noise_power = psd[noise_mask].mean() if noise_mask.any() else 1.0
        snr = float(peak_power / (noise_power + 1e-10))
        bpm = float(peak_freq * 60.0)

        log.debug(
            "rPPG — peak_freq=%.3f Hz (%.1f BPM), SNR=%.2f",
            peak_freq, bpm, snr,
        )
        return snr, bpm

    @timer
    def check(
        self,
        frames: list[np.ndarray],
        bboxes: list[tuple] | None = None,
    ) -> float:
        """
        Run rPPG liveness check on a sequence of frames.

        Args:
            frames: List of BGR full camera frames (NOT aligned crops).
                    Minimum MIN_FRAMES_FOR_RPPG (~60) for reliable detection.
                    More frames = more reliable signal.
            bboxes: Optional list of (x1,y1,x2,y2) face bounding boxes
                    for each frame. Improves ROI accuracy.

        Returns:
            Liveness confidence in [0.0, 1.0]:
              0.0  = no pulse signal detected (likely static image/replay)
              0.5  = insufficient frames for reliable analysis
              0.7+ = clear pulse signal detected (likely live person)
              1.0  = very strong, clean pulse signal

        Note:
            Returns 0.5 (neutral/uncertain) when fewer than min_frames
            are provided rather than 0.0, because absence of signal with
            insufficient data is uninformative — not evidence of spoofing.
        """
        if not frames:
            log.warning("RPPGDetector.check() received empty frames list")
            return 0.0

        if len(frames) < self.min_frames:
            log.info(
                "rPPG: only %d frames (need %d) — returning neutral 0.5",
                len(frames), self.min_frames,
            )
            return 0.5  # neutral — not enough data to decide

        # ── Step 1: Extract green channel signal ──────────────────────────
        green_signal = self._extract_green_signal(frames, bboxes)

        if green_signal.std() < 0.1:
            log.info(
                "rPPG: green signal has near-zero variance (%.4f) — "
                "possible static image",
                green_signal.std(),
            )
            return 0.1

        # ── Step 2: Detrend to remove slow lighting drift ─────────────────
        detrended = scipy_signal.detrend(green_signal)

        # ── Step 3: Bandpass filter to isolate pulse frequencies ──────────
        try:
            filtered = self._bandpass_filter(detrended, self.fps)
        except Exception as e:
            log.warning("rPPG bandpass filter failed: %s", e)
            return 0.5

        # ── Step 4: Compute SNR of pulse peak ─────────────────────────────
        snr, dominant_bpm = self._compute_pulse_snr(filtered, self.fps)

        log.info(
            "rPPG — SNR=%.3f (threshold=%.2f), BPM=%.1f",
            snr, self.min_snr, dominant_bpm,
        )

        # ── Step 5: Score based on SNR ────────────────────────────────────
        if snr < self.min_snr:
            # Weak or absent pulse signal
            score = float(np.clip(snr / self.min_snr * 0.4, 0.0, 0.4))
        else:
            # Clear pulse detected — scale SNR to score
            # SNR of 1.5 → 0.6, SNR of 5.0 → 1.0
            score = float(np.clip(0.4 + (snr - self.min_snr) / 5.0, 0.4, 1.0))

        log.info("RPPGDetector score: %.4f", score)
        return score

    def estimate_heart_rate(self, frames: list[np.ndarray]) -> dict:
        """
        Estimate heart rate from a frame sequence.

        Returns a dict with BPM estimate and confidence for
        display in the HUD or project report.
        """
        if len(frames) < self.min_frames:
            return {"bpm": None, "confidence": 0.0,
                    "error": f"Need {self.min_frames} frames"}

        green_signal = self._extract_green_signal(frames)
        detrended    = scipy_signal.detrend(green_signal)
        filtered     = self._bandpass_filter(detrended, self.fps)
        snr, bpm     = self._compute_pulse_snr(filtered, self.fps)

        return {
            "bpm":        round(bpm, 1),
            "snr":        round(snr, 3),
            "confidence": round(min(1.0, snr / 5.0), 3),
            "valid":      snr >= self.min_snr,
        }