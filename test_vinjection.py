"""
test_video_flash.py
────────────────────────────────────────────────────────────────────────────
Simulates an OBS / Virtual Camera injection attack.
Tests strictly the Active Illumination (Flash) and CNN against a video file.

Run from project root:
    python test_video_flash.py --video "path/to/your/video.mp4"
────────────────────────────────────────────────────────────────────────────
"""

import sys
import cv2
import argparse
import numpy as np

from infrastructure.logger import get_logger
from core_vision.face_detector import FaceDetector
from liveness.active.flash_challenge import FlashChallenge
from liveness.active.reflection_analyzer import ReflectionAnalyzer
from liveness.spatial.cnn_classifier import CNNClassifier

log = get_logger(__name__)

def run_video_flash_test(video_path: str, device: str = "cuda") -> None:
    log.info(f"Starting Video Injection Test on: {video_path}")

    # ── Init ──────────────────────────────────────────────────────────────
    detector   = FaceDetector(device=device)
    challenger = FlashChallenge(
        camera_index=0, # Dummy index, we pass the video cap directly
        flash_duration_ms=250,
        sequence_length=4,
    )
    
    # Using the HSV Glare Trap analyzer
    analyzer   = ReflectionAnalyzer(
        min_alignment=0.30,
        min_colours_passing=3,
    )
    
    cnn        = CNNClassifier(device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video file: {video_path}")
        sys.exit(1)

    window_name = "Video Injection Test | SPACE=start challenge, Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    last_result  = None
    last_verdict = None

    log.info("Playing video. Press SPACE to run flash challenge.")

    # Video playback delay to simulate real-time viewing
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 30.0
    delay_ms = max(1, int(1000 / video_fps))

    while True:
        ret, frame = cap.read()
        
        # Loop the video if it reaches the end before you press SPACE
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        except Exception:
            cv2.putText(annotated, "No face detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)

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
            for color_name, res in breakdown.items():
                if color_name == "summary":
                    continue
                status = "PASS" if res.get("passed") else "FAIL"
                col    = (0, 210, 0) if res.get("passed") else (0, 0, 210)
                
                cv2.putText(
                    annotated,
                    f"  {color_name}: {status} "
                    f"(align={res['alignment']:.2f}, "
                    f"v_shift={res.get('v_shift', 0):.1f})",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                )
                y += 22

            # CNN score
            if last_result and "cnn_score" in last_result:
                cnn_score = last_result['cnn_score']
                cnn_text = f"CNN score: {cnn_score:.3f}"
                if cnn_score < 0.25:
                    cnn_text += " [VETO]"
                cv2.putText(
                    annotated,
                    cnn_text,
                    (20, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                )

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord(" ") and bbox is not None:
            # ── Run flash challenge ────────────────────────────────────────
            log.info("Simulating flash challenge on video stream...")
            cv2.destroyWindow(window_name)

            # We pass the video capture object. The system will "flash" the screen
            # and read the next frames from the video file.
            challenge_result = challenger.run(cap)

            if challenge_result["success"]:
                # Analyse reflection response (Expect this to fail spectacularly)
                is_live, score, breakdown = analyzer.analyze(
                    challenge_result, bbox=bbox
                )

                # Run CNN on the full frames
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
                    "Video Challenge complete — flash_live=%s, "
                    "flash_score=%.3f, cnn_score=%.3f",
                    is_live, score, cnn_score,
                )
            else:
                log.warning("Challenge failed — video ended or not enough frames")
                last_result  = None
                last_verdict = (False, 0.0, {"error": "capture failed"})

            # Reopen window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

    cap.release()
    cv2.destroyAllWindows()
    log.info("Video injection test finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the deepfake video file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    try:
        run_video_flash_test(video_path=args.video, device=args.device)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()