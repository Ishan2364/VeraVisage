"""
test_phase2.py
────────────────────────────────────────────────────────────────────────────
Phase 2 completion test — live webcam demo.

Run from the project root:
    python test_phase2.py

A window opens showing your webcam feed with:
  - Green bounding box around your face
  - 5 coloured landmark dots (eyes, nose, mouth corners)
  - Confidence score in the top-left of the box
  - A small 112×112 aligned crop shown in the top-right corner
  - FPS counter in the top-left of the window

Press Q to quit.

PHASE 2 IS COMPLETE WHEN:
  - The window opens without errors
  - A green box tracks your face as you move
  - The aligned crop in the corner stays upright even when you tilt your head
────────────────────────────────────────────────────────────────────────────
"""

import sys
import time

import cv2
import numpy as np

from infrastructure.logger import get_logger
from core_vision.frame_extractor import stream_webcam
from core_vision.face_detector import FaceDetector
from core_vision.face_aligner import FaceAligner
from infrastructure.exceptions import FaceNotFoundError, MultipleFacesError

log = get_logger(__name__)

# ── Landmark colours (BGR) — one per landmark point ───────────────────────
LANDMARK_COLOURS = [
    (0, 255, 0),    # left eye  — green
    (0, 255, 0),    # right eye — green
    (0, 165, 255),  # nose      — orange
    (255, 0, 0),    # left mouth corner  — blue
    (255, 0, 0),    # right mouth corner — blue
]


def draw_detection(frame: np.ndarray, detection: dict) -> np.ndarray:
    """
    Draw bounding box, landmarks, and confidence onto a frame copy.

    Args:
        frame:     BGR frame, will NOT be modified (we draw on a copy).
        detection: Dict from FaceDetector with bbox, landmarks, confidence.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    bbox = detection["bbox"].astype(int)
    landmarks = detection["landmarks"].astype(int)
    confidence = detection["confidence"]

    x1, y1, x2, y2 = bbox

    # Bounding box
    cv2.rectangle(out, (x1, y1), (x2, y2), color=(0, 220, 0), thickness=2)

    # Confidence label just above the box
    label = f"{confidence:.2f}"
    label_y = max(y1 - 8, 16)
    cv2.putText(
        out, label, (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA,
    )

    # 5 landmark dots
    for (lx, ly), colour in zip(landmarks, LANDMARK_COLOURS):
        cv2.circle(out, (lx, ly), radius=3, color=colour, thickness=-1)

    return out


def embed_aligned_crop(frame: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """
    Embed the 112×112 aligned crop in the top-right corner of the frame.

    Draws a small border around the crop so it's easy to see.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Scale up the 112×112 crop to 2× for visibility
    display_size = 224
    crop_resized = cv2.resize(crop, (display_size, display_size))

    # Position: top-right with 10px margin
    x_offset = w - display_size - 10
    y_offset = 10

    # Paste the crop
    out[y_offset:y_offset + display_size, x_offset:x_offset + display_size] = crop_resized

    # White border around the crop
    cv2.rectangle(
        out,
        (x_offset - 1, y_offset - 1),
        (x_offset + display_size, y_offset + display_size),
        color=(255, 255, 255), thickness=1,
    )

    # Label
    cv2.putText(
        out, "aligned crop", (x_offset, y_offset + display_size + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return out


def run_demo(device: str = "cuda", camera_index: int = 0) -> None:
    """
    Main loop — opens webcam, runs detection + alignment every frame,
    draws results, shows live window.
    """
    log.info("Starting Phase 2 live demo — device=%s, camera=%d", device, camera_index)

    detector = FaceDetector(device=device, confidence_threshold=0.5)
    aligner = FaceAligner()

    # FPS tracking
    fps_history: list[float] = []
    prev_time = time.perf_counter()

    window_name = "VeraVisage — Phase 2 | Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    log.info("Window opened — move your face in front of the camera")

    for frame in stream_webcam(camera_index=camera_index):

        annotated = frame.copy()

        # ── Detection ──────────────────────────────────────────────────────
        try:
            faces = detector.detect(frame)

            # ── Detection ──────────────────────────────────────────────────────
            try:
                best_face = detector.detect_single(frame)
                annotated = draw_detection(annotated, best_face)
                aligned_crop = aligner.align_from_detection(frame, best_face)
                annotated = embed_aligned_crop(annotated, aligned_crop)

            except MultipleFacesError:
                cv2.putText(
                    annotated,
                    "Multiple faces detected — one person only",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 100, 255), 2, cv2.LINE_AA,
                )

            except FaceNotFoundError:
                cv2.putText(
                    annotated,
                    "No face detected — look at the camera",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 100, 255), 2, cv2.LINE_AA,
                )

            except Exception as e:
                log.warning("Detection error on this frame: %s", e)

            else:
                # No face — show a prompt
                cv2.putText(
                    annotated,
                    "No face detected — look at the camera",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA,
                )

        except Exception as e:
            log.warning("Detection error on this frame: %s", e)

        # ── FPS counter ────────────────────────────────────────────────────
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # ── Phase label ────────────────────────────────────────────────────
        cv2.putText(
            annotated,
            "VeraVisage  Phase 2 — detection + alignment",
            (12, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
        )

        cv2.imshow(window_name, annotated)

        # Q or Escape to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            log.info("Quit key pressed — closing demo")
            break

    cv2.destroyAllWindows()
    log.info("Phase 2 demo finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VeraVisage Phase 2 — live face detection demo")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (default: cuda)")
    parser.add_argument("--camera", default=0, type=int,
                        help="Webcam index (default: 0)")
    args = parser.parse_args()

    try:
        run_demo(device=args.device, camera_index=args.camera)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)