"""
pipeline/enroll_pipeline.py
────────────────────────────────────────────────────────────────────────────
End-to-end enrolment pipeline:
  Camera → Detect → CNN liveness gate → Align → Embed (30 frames) → Store

ENROLMENT FLOW
──────────────
1. Open webcam, show countdown
2. Capture 30 frames of the user's face
3. Run CNN liveness gate — reject if MiniFASNet says spoof
   (We don't run the full flash challenge for enrolment — it would be
   annoying UX to do a flash challenge every time someone wants to enrol.
   CNN alone is sufficient here since the enrolment operator can visually
   verify the person is present.)
4. Detect and align every frame → 30 aligned 112×112 crops
5. Generate ArcFace embedding per crop
6. Average all embeddings → single robust stored vector
7. Save to identity_store with user_id

NOTE ON LIVENESS DURING ENROLMENT
────────────────────────────────────
Full dual-veto liveness (CNN + Flash) runs during AUTHENTICATION.
Enrolment only runs the CNN gate — this is intentional and standard
practice. The assumption is that enrolment happens in a controlled
environment (the user is setting up their own account) while
authentication is the security-critical moment.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from core_vision.face_detector import FaceDetector
from core_vision.face_aligner import FaceAligner
from infrastructure.exceptions import FaceNotFoundError, EnrolmentError
from infrastructure.logger import get_logger
from liveness.spatial.cnn_classifier import CNNClassifier
from core_vision.embedding_generator import EmbeddingGenerator
from verification.identity_store import IdentityStore

log = get_logger(__name__)

# Number of frames to capture and average for enrolment
ENROL_FRAMES    = 30
# Minimum CNN score to allow enrolment
CNN_ENROL_GATE  = 0.25
# Minimum successfully aligned frames to proceed
MIN_GOOD_FRAMES = 15


def run_enroll(
    user_id: str,
    device: str = "cuda",
    camera_index: int = 0,
    show_preview: bool = True,
) -> dict:
    """
    Enrol a new user by capturing their face from the webcam.

    Args:
        user_id:       Unique identifier for this user.
                       Must be alphanumeric (used as filename).
        device:        "cuda" or "cpu" for model inference.
        camera_index:  Webcam index (0 = default).
        show_preview:  If True, opens a window showing the camera feed
                       and progress during enrolment.

    Returns:
        Status dict:
        {
            "user_id":      str,
            "status":       "enrolled" | "failed",
            "frames_used":  int,
            "reason":       str,
        }
    """
    log.info("Starting enrolment for user '%s'", user_id)

    # ── Init modules ──────────────────────────────────────────────────────
    detector  = FaceDetector(device=device)
    aligner   = FaceAligner()
    cnn       = CNNClassifier(device=device)
    embedder  = EmbeddingGenerator(device=device)
    store     = IdentityStore()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "user_id": user_id,
            "status":  "failed",
            "frames_used": 0,
            "reason":  f"Cannot open camera {camera_index}",
        }

    # Warm up camera
    for _ in range(10):
        cap.read()

    window_name = f"VeraVisage — Enrolling: {user_id} | Q=cancel"

    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 500)

    aligned_frames: list[np.ndarray] = []
    cnn_scores:     list[float]       = []
    frame_count     = 0
    collection_done = False

    # Countdown before capture starts
    countdown_start = time.perf_counter()
    COUNTDOWN_SECS  = 3

    log.info(
        "Enrolment capturing %d frames — showing %d second countdown",
        ENROL_FRAMES, COUNTDOWN_SECS,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated    = frame.copy()
            elapsed      = time.perf_counter() - countdown_start

            # ── Countdown phase ───────────────────────────────────────────
            if elapsed < COUNTDOWN_SECS:
                remaining = COUNTDOWN_SECS - int(elapsed)
                cv2.putText(
                    annotated,
                    f"Look at the camera — {remaining}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 210, 255), 2,
                )
                cv2.putText(
                    annotated,
                    f"Enrolling: {user_id}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1,
                )

            # ── Capture phase ─────────────────────────────────────────────
            elif not collection_done:
                try:
                    face    = detector.detect_single(frame)
                    aligned = aligner.align_from_detection(frame, face)
                    bbox    = face["bbox"].tolist()

                    # Draw box
                    x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                  (0, 210, 0), 2)

                    # CNN liveness gate on full frame
                    if frame_count % 5 == 0:
                        cnn_score = cnn.check_with_bbox([frame], [bbox])
                        cnn_scores.append(cnn_score)

                    aligned_frames.append(aligned)

                    # Progress bar
                    progress = len(aligned_frames) / ENROL_FRAMES
                    bar_w    = int(progress * 400)
                    cv2.rectangle(annotated,
                                  (20, annotated.shape[0] - 40),
                                  (20 + bar_w, annotated.shape[0] - 20),
                                  (0, 210, 0), -1)
                    cv2.rectangle(annotated,
                                  (20, annotated.shape[0] - 40),
                                  (420, annotated.shape[0] - 20),
                                  (100, 100, 100), 1)
                    cv2.putText(
                        annotated,
                        f"Capturing... {len(aligned_frames)}/{ENROL_FRAMES}",
                        (20, annotated.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1,
                    )

                    if len(aligned_frames) >= ENROL_FRAMES:
                        collection_done = True

                except FaceNotFoundError:
                    cv2.putText(annotated, "Keep face in frame",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 100, 255), 2)
                except Exception as e:
                    log.warning("Frame %d error: %s", frame_count, e)

            # ── Done — process and exit ───────────────────────────────────
            else:
                cv2.putText(annotated, "Processing...",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 210, 255), 2)
                if show_preview:
                    cv2.imshow(window_name, annotated)
                    cv2.waitKey(500)
                break

            if show_preview:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    log.info("Enrolment cancelled by user")
                    cap.release()
                    cv2.destroyAllWindows()
                    return {
                        "user_id":     user_id,
                        "status":      "failed",
                        "frames_used": 0,
                        "reason":      "Cancelled by user",
                    }

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    # ── Post-capture checks ───────────────────────────────────────────────
    if len(aligned_frames) < MIN_GOOD_FRAMES:
        return {
            "user_id":     user_id,
            "status":      "failed",
            "frames_used": len(aligned_frames),
            "reason": (
                f"Only {len(aligned_frames)} good frames captured "
                f"(need {MIN_GOOD_FRAMES}). "
                f"Ensure face is clearly visible and well-lit."
            ),
        }

    # CNN liveness gate — reject if MiniFASNet consistently says spoof
    if cnn_scores:
        mean_cnn = float(np.mean(cnn_scores))
        log.info("Enrolment CNN mean score: %.3f", mean_cnn)
        if mean_cnn < CNN_ENROL_GATE:
            return {
                "user_id":     user_id,
                "status":      "failed",
                "frames_used": len(aligned_frames),
                "reason": (
                    f"CNN liveness gate failed (score={mean_cnn:.3f} "
                    f"< {CNN_ENROL_GATE}). "
                    f"Possible spoofing attempt during enrolment."
                ),
            }

    # ── Generate embedding ────────────────────────────────────────────────
    try:
        embedding = embedder.generate_batch(aligned_frames)
    except Exception as e:
        return {
            "user_id":     user_id,
            "status":      "failed",
            "frames_used": len(aligned_frames),
            "reason":      f"Embedding generation failed: {e}",
        }

    # ── Store embedding ───────────────────────────────────────────────────
    try:
        store.enrol(user_id, embedding)
    except Exception as e:
        return {
            "user_id":     user_id,
            "status":      "failed",
            "frames_used": len(aligned_frames),
            "reason":      f"Storage failed: {e}",
        }

    log.info(
        "Enrolment successful — user='%s', frames=%d",
        user_id, len(aligned_frames),
    )

    return {
        "user_id":     user_id,
        "status":      "enrolled",
        "frames_used": len(aligned_frames),
        "reason":      f"Successfully enrolled with {len(aligned_frames)} frames.",
    }