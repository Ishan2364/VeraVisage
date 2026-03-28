"""
test_flash_challenge.py
────────────────────────────────────────────────────────────────────────────
Standalone test for the Active Illumination Challenge-Response system.

Run from project root:
    python test_flash_challenge.py

What happens:
  1. Webcam opens, shows camera feed
  2. Press SPACE to start the flash challenge
  3. Screen flashes 3 random colours (250ms each)
  4. System analyses your skin colour response
  5. Shows LIVE or SPOOF verdict with breakdown

Test scenarios:
  - Your real face        → should show LIVE
  - Phone photo           → should show SPOOF (screen shifts uniformly)
  - OBS virtual camera    → should show SPOOF (no response to flash)
  - Printed photo         → should show SPOOF (paper doesn't reflect)
────────────────────────────────────────────────────────────────────────────
"""

import sys
import time

import cv2
import numpy as np

from infrastructure.logger import get_logger
from core_vision.face_detector import FaceDetector
from core_vision.face_aligner import FaceAligner
from liveness.active.flash_challenge import FlashChallenge
from liveness.active.reflection_analyzer import ReflectionAnalyzer
from liveness.spatial.cnn_classifier import CNNClassifier

log = get_logger(__name__)


def run_flash_test(
    device: str = "cuda",
    camera_index: int = 0,
) -> None:

    log.info("Flash challenge test starting")

    # ── Init ──────────────────────────────────────────────────────────────
    detector   = FaceDetector(device=device)
    aligner    = FaceAligner()
    challenger = FlashChallenge(
        camera_index=camera_index,
        flash_duration_ms=250,
        sequence_length=3,
    )
    
    # FIXED: Removed the deprecated min_magnitude=3.0 argument
    analyzer   = ReflectionAnalyzer(
        min_alignment=0.30,
        min_colours_passing=2,
    )
    
    cnn        = CNNClassifier(device=device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera %d", camera_index)
        sys.exit(1)

    # Warm up camera
    for _ in range(10):
        cap.read()

    window_name = "VeraVisage — Flash Challenge Test | SPACE=start, Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    last_result  = None
    last_verdict = None

    log.info("Press SPACE to run flash challenge, Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ── Detect face ───────────────────────────────────────────────────
        bbox = None
        try:
            face = detector.detect_single(frame)
            bbox = face["bbox"].tolist()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 0), 2)
            cv2.putText(annotated, f"{face['confidence']:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 210, 0), 1)
        except Exception:
            cv2.putText(annotated, "No face detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)

        # ── Instructions ──────────────────────────────────────────────────
        cv2.putText(annotated,
                    "Press SPACE to run flash challenge",
                    (20, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1)

        # ── Show last result ───────────────────────────────────────────────
        if last_verdict is not None:
            is_live, score, breakdown = last_verdict
            v_color = (0, 210, 0) if is_live else (0, 0, 210)
            verdict = "LIVE" if is_live else "SPOOF"

            cv2.putText(annotated, f"Flash result: {verdict}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, v_color, 2)
            cv2.putText(annotated, f"Score: {score:.3f}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, v_color, 1)

            # Per-colour breakdown
            y = 120
            summary = breakdown.get("summary", {})
            for color_name, res in breakdown.items():
                if color_name == "summary":
                    continue
                status = "PASS" if res.get("passed") else "FAIL"
                col    = (0, 210, 0) if res.get("passed") else (0, 0, 210)
                
                # FIXED: Updated 'magnitude' key to 'v_shift'
                cv2.putText(
                    annotated,
                    f"  {color_name}: {status} "
                    f"(align={res['alignment']:.2f}, "
                    f"v_shift={res.get('v_shift', 0):.1f})",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                )
                y += 22

            # CNN score if available
            if last_result and "cnn_score" in last_result:
                cv2.putText(
                    annotated,
                    f"CNN score: {last_result['cnn_score']:.3f}",
                    (20, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                )

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord(" ") and bbox is not None:
            # ── Run flash challenge ────────────────────────────────────────
            log.info("Starting flash challenge...")
            cv2.destroyWindow(window_name)

            challenge_result = challenger.run(cap)

            if challenge_result["success"]:
                # Analyse reflection response
                is_live, score, breakdown = analyzer.analyze(
                    challenge_result, bbox=bbox
                )

                # Also run CNN on baseline frames for combined verdict
                # FIXED: Cleaned up the try/except block. Sends full frames to CNN.
                baseline_frames = challenge_result["baseline_frames"]
                try:
                    cnn_score = cnn.check_with_bbox(
                        baseline_frames[:5],
                        [bbox] * len(baseline_frames[:5]),
                    )
                except Exception as e:
                    log.warning("CNN scoring error: %s", e)
                    cnn_score = 0.5

                last_result  = {"cnn_score": cnn_score}
                last_verdict = (is_live, score, breakdown)

                log.info(
                    "Challenge complete — flash_live=%s, "
                    "flash_score=%.3f, cnn_score=%.3f",
                    is_live, score, cnn_score,
                )
            else:
                log.warning("Challenge failed — not enough frames captured")
                last_result  = None
                last_verdict = (False, 0.0, {"error": "capture failed"})

            # Reopen camera feed window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

    cap.release()
    cv2.destroyAllWindows()
    log.info("Flash challenge test finished")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()

    try:
        run_flash_test(device=args.device, camera_index=args.camera)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()