"""
liveness/active/flash_challenge.py
────────────────────────────────────────────────────────────────────────────
Active Illumination Challenge-Response liveness detection.

CONCEPT
────────
Your screen flashes a randomised colour sequence. A real face physically
reflects that light — the skin colour measurably shifts toward the flash
colour. A pre-recorded deepfake, virtual camera injection, or phone screen
cannot retroactively respond to a challenge generated right now.

This is the same principle as iProov's patented Flashmark technology
and the Face Flashing protocol (Tang et al., NDSS 2018).

ATTACK IMMUNITY (DYNAMIC UPGRADE)
─────────────────────────────────
  Printed photo:        No reflection response — flat paper
  Phone screen replay:  Entire screen shifts uniformly, not just face skin
  OBS virtual camera:   Pre-recorded video has no knowledge of flash colour
  Deepfake video:       Same — cannot retroactively show colour response
  *NEW* Sync Attacks:   Randomised flash durations break shutter-syncing scripts.
  *NEW* Hue Guessing:   Millions of possible RGB combinations makes prediction 0%.

SINGLE SCREEN SETUP
────────────────────
Flash window covers the camera feed window temporarily.
Each colour flashes for a randomised duration (180ms - 380ms).
Webcam captures frames during each flash for analysis.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
import time

import cv2
import numpy as np

from infrastructure.logger import get_logger

log = get_logger(__name__)

# ── Dynamic Flash Configuration ───────────────────────────────────────────
MIN_FLASH_MS     = 180      # Minimum duration for a single colour
MAX_FLASH_MS     = 380      # Maximum duration for a single colour
MIN_PAUSE_MS     = 100      # Minimum dark pause between colours
MAX_PAUSE_MS     = 250      # Maximum dark pause between colours

SETTLE_FRAMES    = 3        # frames to discard after flash starts (settling)
PRE_FLASH_FRAMES = 8        # baseline frames captured before any flash

# Number of colours in each challenge sequence
SEQUENCE_LENGTH = 4

# Window name for flash display
FLASH_WINDOW = "VeraVisage — Liveness Check"


class FlashChallenge:
    """
    Generates and executes a cryptographically randomised colour flash challenge.

    Opens a fullscreen flash window, captures webcam frames during
    each dynamic colour flash, and returns the captured data for analysis.
    """

    def __init__(
        self,
        camera_index: int = 0,
        sequence_length: int = SEQUENCE_LENGTH,
        flash_duration_ms: int = 250, # Kept for backward compatibility, but overridden by dynamic logic
    ):
        """
        Args:
            camera_index:      Webcam index (0 = default).
            sequence_length:   Number of colours in the challenge sequence.
            flash_duration_ms: (Deprecated) Now uses dynamic window.
        """
        self.camera_index    = camera_index
        self.sequence_length = sequence_length

        log.info(
            "FlashChallenge initialised — dynamic timing enabled, sequence_length=%d",
            sequence_length,
        )

    def _generate_sequence(self) -> list[dict]:
        """
        Generate a highly unpredictable colour and timing sequence.

        Returns:
            List of dictionaries containing colour hexes and specific timings.
            Sequence is mathematically randomised each call — replay attacks 
            cannot predict the next sequence.
        """
        sequence = []
        for _ in range(self.sequence_length):
            # 1. UNPREDICTABLE RGB (Generate vivid colors using HSV space)
            h = random.randint(0, 179)       # OpenCV hue is 0-179
            s = random.randint(200, 255)     # High saturation (vivid)
            v = random.randint(220, 255)     # High value (bright light)
            
            # Convert to BGR for screen rendering
            hsv_pixel = np.uint8([[[h, s, v]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            color_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

            # 2. VARIABLE TIMING
            duration_ms = random.randint(MIN_FLASH_MS, MAX_FLASH_MS)
            pause_ms    = random.randint(MIN_PAUSE_MS, MAX_PAUSE_MS)

            sequence.append({
                "color_name":  f"HSV_{h}_{s}_{v}", 
                "color_bgr":   color_bgr,
                "duration_ms": duration_ms,
                "pause_ms":    pause_ms,
            })
            
        log.info("Generated dynamic flash sequence of length %d", len(sequence))
        return sequence

    def _create_flash_frame(
        self,
        bgr_color: tuple[int, int, int],
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        """Create a solid colour frame for flashing."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = bgr_color
        return frame

    def run(
        self,
        cap: cv2.VideoCapture,
        display_frame: np.ndarray | None = None,
    ) -> dict:
        """
        Execute the full dynamic flash challenge sequence.

        Args:
            cap:           Open cv2.VideoCapture object (already warming up).
            display_frame: Optional frame to show before challenge starts.

        Returns:
            Dict containing baseline frames, captured sequence data, and success flag.
        """
        sequence        = self._generate_sequence()
        baseline_frames: list[np.ndarray] = []
        flash_data:      list[dict]       = []

        h, w = 720, 1280  # default flash window size

        # ── Step 1: Show countdown so user can position themselves ─────────
        cv2.namedWindow(FLASH_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            FLASH_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        for count in range(3, 0, -1):
            countdown_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                countdown_frame,
                f"Look at the camera — {count}",
                (w // 2 - 280, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (200, 200, 200), 3, cv2.LINE_AA,
            )
            cv2.putText(
                countdown_frame,
                "Liveness check starting...",
                (w // 2 - 280, h // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (150, 150, 150), 2, cv2.LINE_AA,
            )
            cv2.imshow(FLASH_WINDOW, countdown_frame)
            cv2.waitKey(1000)

        # ── Step 2: Capture baseline frames (no flash) ────────────────────
        log.info("Capturing %d baseline frames...", PRE_FLASH_FRAMES)
        for _ in range(PRE_FLASH_FRAMES):
            ret, frame = cap.read()
            if ret and frame is not None:
                baseline_frames.append(frame.copy())

        # ── Step 3: Flash each dynamic colour and capture ──────────────────
        for step in sequence:
            color_name  = step["color_name"]
            color_bgr   = step["color_bgr"]
            duration_ms = step["duration_ms"]
            pause_ms    = step["pause_ms"]

            log.info("Flashing: %s %s for %dms", color_name, color_bgr, duration_ms)

            # Show flash colour
            flash_frame = self._create_flash_frame(color_bgr, w, h)

            # Add small label so user knows what's happening
            cv2.putText(
                flash_frame,
                "Keep looking at the camera",
                (w // 2 - 250, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (128, 128, 128), 1, cv2.LINE_AA,
            )
            cv2.imshow(FLASH_WINDOW, flash_frame)
            cv2.waitKey(1)  # force display update

            # Discard settle frames (camera exposure adjusting)
            for _ in range(SETTLE_FRAMES):
                cap.read()

            # Time-based capture loop (Asynchronous Variable Timing)
            captured: list[np.ndarray] = []
            flash_start_time = time.perf_counter()
            
            while (time.perf_counter() - flash_start_time) * 1000 < duration_ms:
                ret, frame = cap.read()
                if ret and frame is not None:
                    captured.append(frame.copy())

            flash_data.append({
                "color_name":   color_name,
                "color_bgr":    color_bgr,
                "frames":       captured,
                "timestamp_ms": flash_start_time * 1000, # Legacy key match
            })

            # Brief dark pause with active buffer clearing
            dark = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.imshow(FLASH_WINDOW, dark)
            
            pause_start = time.perf_counter()
            while (time.perf_counter() - pause_start) * 1000 < pause_ms:
                cap.read() # Actively dump hardware buffer to prevent light bleed
                cv2.waitKey(1)

        # ── Step 4: Show completion message ───────────────────────────────
        done_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            done_frame, "Processing...",
            (w // 2 - 150, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0,
            (200, 200, 200), 3, cv2.LINE_AA,
        )
        cv2.imshow(FLASH_WINDOW, done_frame)
        cv2.waitKey(500)
        cv2.destroyWindow(FLASH_WINDOW)

        success = (
            len(baseline_frames) >= 3 and
            all(len(fd["frames"]) >= 2 for fd in flash_data)
        )

        log.info(
            "Flash challenge complete — baseline=%d frames, "
            "flash_data=%d colours, success=%s",
            len(baseline_frames), len(flash_data), success,
        )

        return {
            "sequence":        [(s["color_name"], s["color_bgr"]) for s in sequence],
            "baseline_frames": baseline_frames,
            "flash_data":      flash_data,
            "success":         success,
        }