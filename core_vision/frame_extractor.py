"""
core_vision/frame_extractor.py
────────────────────────────────────────────────────────────────────────────
Handles all video → frame conversion for the deepfake_auth pipeline.

TWO MODES
─────────
1. Live webcam  — capture_live_frames()
   Opens the default camera, captures N frames with a small delay between
   each to let the auto-exposure settle. Used during authentication.

2. Video file   — extract_frames_from_video()
   Reads a pre-recorded video and yields frames sampled at a target FPS.
   Used during enrolment (if the user uploads a short video clip) and
   during evaluation / testing.

DESIGN DECISION
───────────────
Both functions return BGR NumPy arrays — OpenCV's native format.
Nothing in this module converts to RGB. The conversion happens at the
boundary with PyTorch/insightface inside face_detector.py, keeping the
responsibility clear.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from infrastructure.exceptions import FaceNotFoundError
from infrastructure.logger import get_logger
from infrastructure.utils import timer

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Live webcam capture
# ══════════════════════════════════════════════════════════════════════════

@timer
def capture_live_frames(
    num_frames: int = 20,
    camera_index: int = 0,
    warmup_frames: int = 5,
    inter_frame_delay: float = 0.05,
) -> list[np.ndarray]:
    """
    Open the webcam and capture a fixed number of BGR frames.

    Args:
        num_frames:        How many frames to capture for the pipeline.
                           20 gives the liveness checks enough temporal
                           signal without making the user wait too long.
        camera_index:      OpenCV camera index. 0 = default/built-in webcam.
                           Try 1 or 2 if you have multiple cameras.
        warmup_frames:     Frames to discard at the start while the camera
                           auto-exposure is stabilising. Without this the
                           first few frames are often dark or blown out.
        inter_frame_delay: Seconds to wait between captured frames.
                           0.05s = ~20 fps capture rate.

    Returns:
        List of np.ndarray, each shape (H, W, 3), dtype uint8, BGR.

    Raises:
        RuntimeError:      If the camera cannot be opened.
        FaceNotFoundError: (downstream) — not raised here, but callers
                           should expect it from face_detector.py.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            f"Check that your webcam is connected and not in use by another app."
        )

    log.info(
        "Camera opened — index=%d, warming up with %d discarded frames",
        camera_index,
        warmup_frames,
    )

    try:
        # ── Warmup: let auto-exposure settle ──────────────────────────────
        for _ in range(warmup_frames):
            cap.read()  # discard frame

        # ── Capture the real frames ───────────────────────────────────────
        frames: list[np.ndarray] = []

        for i in range(num_frames):
            ret, frame = cap.read()

            if not ret or frame is None:
                log.warning("Frame %d/%d could not be read — skipping", i + 1, num_frames)
                continue

            frames.append(frame)
            log.debug("Captured frame %d/%d  shape=%s", i + 1, num_frames, frame.shape)

            if inter_frame_delay > 0:
                time.sleep(inter_frame_delay)

        if not frames:
            raise RuntimeError(
                "Camera opened but zero frames were captured. "
                "Try a different camera_index or check your webcam drivers."
            )

        log.info("Captured %d frames from camera %d", len(frames), camera_index)
        return frames

    finally:
        # Always release the camera, even if an exception was raised
        cap.release()
        log.debug("Camera %d released", camera_index)


# ══════════════════════════════════════════════════════════════════════════
# Video file extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_frames_from_video(
    path: str | Path,
    target_fps: int = 5,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames sampled at target_fps from a video file.

    We sample at a lower FPS than the source video because:
      - A 30fps video has many near-identical consecutive frames
      - Liveness checks need temporal diversity, not raw frame count
      - Lower frame count = faster pipeline = better UX

    Args:
        path:       Path to any video file OpenCV can read
                    (.mp4, .avi, .mov, .mkv, etc.)
        target_fps: How many frames per second of video to yield.
                    5 means one frame every 200ms of video time.

    Yields:
        np.ndarray — one BGR frame at a time, shape (H, W, 3).

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError:      If OpenCV cannot open the video.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    # Work out the sampling interval in terms of frame indices
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, int(source_fps / target_fps))

    log.info(
        "Extracting from %s — source_fps=%.1f, target_fps=%d, "
        "sample_every=%d_frames, total_source_frames=%d",
        path.name, source_fps, target_fps, sample_every, total_frames,
    )

    frame_idx = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # End of video

            if frame_idx % sample_every == 0:
                log.debug("Yielding frame %d (source index %d)", yielded, frame_idx)
                yield frame
                yielded += 1

            frame_idx += 1

    finally:
        cap.release()

    log.info("Extracted %d frames from %s", yielded, path.name)


# ══════════════════════════════════════════════════════════════════════════
# Live preview (used by test_phase2.py)
# ══════════════════════════════════════════════════════════════════════════

def stream_webcam(camera_index: int = 0) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from the webcam indefinitely until the user quits.

    This is used by the live demo / test script.
    The caller is responsible for breaking the loop (e.g. on 'q' keypress).

    Yields:
        np.ndarray — live BGR frame, shape (H, W, 3).

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    log.info("Streaming webcam %d — press Q in the window to quit", camera_index)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame from camera %d", camera_index)
                break
            yield frame
    finally:
        cap.release()
        log.debug("Webcam stream closed")