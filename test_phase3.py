"""
test_phase3.py
────────────────────────────────────────────────────────────────────────────
Phase 3 Final Liveness Demo — Spatial (CNN) + Physical (Flash Challenge)

ARCHITECTURE: DUAL-VETO FUSION
──────────────────────────────
This pipeline strips out brittle passive checks (Blinks/Flow) in favour of 
unfakeable physical interaction. 

Guard 1: The Spatial CNN (MiniFASNet)
  - Looks at the full frame. 
  - Vetoes if it sees phone bezels, paper edges, or deepfake artifacts.
  
Guard 2: The HSV Flash Challenge
  - Flashes 4 random colours.
  - Vetoes if the surface is too flat (Phone/Paper) or too glossy (Glass).
  - Vetoes if the target doesn't respond to light at all (Video Injection).

Fusion Logic:
  If CNN < 0.25 -> SPOOF (Spatial Veto)
  If Flash Fails -> SPOOF (Physical Veto)
  If Both Pass -> Weighted Score (40% CNN + 60% Flash)

Run:
    python test_phase3.py
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np

from infrastructure.logger import get_logger
from infrastructure.exceptions import FaceNotFoundError, MultipleFacesError
from core_vision.face_detector import FaceDetector
from core_vision.face_aligner import FaceAligner
from core_vision.frame_extractor import stream_webcam
from liveness.active.flash_challenge import FlashChallenge
from liveness.active.reflection_analyzer import ReflectionAnalyzer
from liveness.spatial.cnn_classifier import CNNClassifier

log = get_logger(__name__)

# ── Dual Veto Configuration ───────────────────────────────────────────────
CNN_VETO_GATE   = 0.25
FLASH_VETO_GATE = 0.50
THRESHOLD       = 0.45


class LivenessHUD:
    """Tracks state and renders the new Dual-Veto HUD overlay."""

    def __init__(self):
        self.cnn_score    = None
        self.flash_result = None
        self.final_score  = 0.0
        self.verdict      = "AWAITING FLASH"
        self.verdict_msg  = "Press SPACE to prove liveness"
        
        self.face_is_stable = True
        self._bbox_history  = deque(maxlen=10)

        # Background CNN thread
        self._cnn_lock    = threading.Lock()
        self._cnn_running = False

    def update_bbox(self, bbox: np.ndarray) -> None:
        """Track bbox centre to ensure the user isn't shaking the camera."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._bbox_history.append((cx, cy))
        if len(self._bbox_history) >= 5:
            positions = np.array(list(self._bbox_history))
            movement  = np.std(positions, axis=0).mean()
            self.face_is_stable = movement < 25.0
        else:
            self.face_is_stable = True

    def request_cnn_check(
        self,
        classifier: CNNClassifier,
        full_frames: list[np.ndarray],
        bboxes: list,
    ) -> None:
        """Run CNN check in background thread using full frames."""
        with self._cnn_lock:
            if self._cnn_running:
                return
            self._cnn_running = True

        def _run():
            try:
                score = classifier.check_with_bbox(full_frames, bboxes)
                with self._cnn_lock:
                    self.cnn_score = score
            except Exception as e:
                log.warning("CNN check error: %s", e)
            finally:
                with self._cnn_lock:
                    self._cnn_running = False

        threading.Thread(target=_run, daemon=True).start()

    def compute_fusion(self) -> None:
        """Computes the strict Dual-Veto fusion with State Locking."""
        # 1. Default to the CNN score while waiting for the user to press SPACE
        cnn = self.cnn_score if self.cnn_score is not None else 0.5
        
        # PRE-FLASH STATE
        if not self.flash_result:
            self.final_score = cnn
            self.verdict = "AWAITING FLASH"
            
            if cnn < CNN_VETO_GATE:
                self.verdict = "SPOOF"
                self.verdict_msg = "Spatial Artifacts Detected (CNN Veto)"
            else:
                self.verdict_msg = "Press SPACE to start Flash Challenge"
            return

        # POST-FLASH STATE
        flash_score   = self.flash_result.get("score", 0.0)
        flash_is_live = self.flash_result.get("is_live", False)

        # GATE 1: CNN Hard Veto
        # (We use the CNN score captured right BEFORE the flash started)
        if cnn < CNN_VETO_GATE:
            self.final_score = 0.0
            self.verdict = "SPOOF"
            self.verdict_msg = "Spatial Artifacts Detected (CNN Veto)"
            return

        # GATE 2: Flash Hard Veto
        if not flash_is_live or flash_score < FLASH_VETO_GATE:
            self.final_score = 0.0
            self.verdict = "SPOOF"
            self.verdict_msg = "Physical Reflection Failed (Flash Veto)"
            return

        # GATE 3: Both Pass. Compute weighted confidence.
        # This math only runs if CNN >= 0.25 AND Flash >= 0.50
        fused = (0.40 * cnn) + (0.60 * flash_score)
        self.final_score = float(np.clip(fused, 0.0, 1.0))
        
        if self.final_score >= THRESHOLD:
            self.verdict = "LIVE"
            self.verdict_msg = "Verified Physical Human"
        else:
            # Mathematical fail-safe: just in case the weights result in a low score
            self.verdict = "SPOOF"
            self.verdict_msg = "Low Combined Confidence"

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw Dual-Veto HUD onto frame copy."""
        out  = frame.copy()
        h, w = out.shape[:2]

        # Background panel
        overlay = out.copy()
        cv2.rectangle(overlay, (10, h - 190), (450, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, out, 0.40, 0, out)

        y = h - 160

        # CNN Score
        if self.cnn_score is None:
            cnn_str = "CNN Guard:   analysing..."
            cnn_col = (150, 150, 150)
        else:
            veto = self.cnn_score < CNN_VETO_GATE
            cnn_str = f"CNN Guard:   {self.cnn_score:.3f}" + (" [VETO]" if veto else " [PASS]")
            cnn_col = (0, 0, 220) if veto else (0, 210, 0)
        cv2.putText(out, cnn_str, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cnn_col, 1)
        y += 25

        # Flash Score
        if self.flash_result is None:
            flash_str = "Flash Guard: waiting..."
            flash_col = (150, 150, 150)
        else:
            f_score = self.flash_result.get("score", 0.0)
            veto = not self.flash_result.get("is_live", False)
            flash_str = f"Flash Guard: {f_score:.3f}" + (" [VETO]" if veto else " [PASS]")
            flash_col = (0, 0, 220) if veto else (0, 210, 0)
        cv2.putText(out, flash_str, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, flash_col, 1)
        y += 30

        # Fused score bar
        cv2.putText(out, f"Final Score: {self.final_score:.3f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 18
        bar_w   = int(self.final_score * 300)
        bar_col = (0, 210, 0) if self.final_score >= THRESHOLD else (0, 100, 255)
        cv2.rectangle(out, (20, y), (20 + bar_w, y + 10), bar_col, -1)
        cv2.rectangle(out, (20, y), (320, y + 10), (100, 100, 100), 1)
        
        # Threshold tick
        tx = 20 + int(THRESHOLD * 300)
        cv2.rectangle(out, (tx-1, y-2), (tx+1, y+12), (255, 255, 255), -1)
        y += 30

        # Final Verdict
        v_col = (0, 210, 0) if self.verdict == "LIVE" else \
                (0, 200, 255) if self.verdict == "AWAITING FLASH" else (0, 0, 220)
        cv2.putText(out, self.verdict, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, v_col, 2)
        cv2.putText(out, f"- {self.verdict_msg}", (110 if self.verdict == "LIVE" else 130, y - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return out


def run_demo(device: str = "cuda", camera_index: int = 0) -> None:

    log.info("Phase 3 — Dual-Veto Liveness Demo (CNN + HSV Flash)")

    # ── Init ──────────────────────────────────────────────────────────────
    detector   = FaceDetector(device=device)
    classifier = CNNClassifier(device=device)
    
    challenger = FlashChallenge(
        camera_index=camera_index,
        flash_duration_ms=250,
        sequence_length=4,      # 4 colours for mathematical security
    )
    
    analyzer   = ReflectionAnalyzer(
        min_alignment=0.30,
        min_colours_passing=3,  # Needs 3/4 colours to pass physics
    )
    
    hud = LivenessHUD()

    # Buffers for CNN
    frame_buffer = deque(maxlen=30)
    bbox_buffer  = deque(maxlen=30)

    fps_history = deque(maxlen=30)
    prev_time   = time.perf_counter()
    frame_count = 0

    window_name = "VeraVisage — Dual Veto Final | SPACE=Flash, Q=Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Use the generator directly so we have access to the capture object
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera.")
        sys.exit(1)
        
    for _ in range(10): cap.read() # Warmup

    log.info("Running. Press SPACE to trigger the Flash Challenge.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        annotated = frame.copy()

        # ── Detect Face ───────────────────────────────────────────────────
        bbox = None
        try:
            face = detector.detect_single(frame)
            bbox = face["bbox"].tolist()
            hud.update_bbox(face["bbox"])
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)

            frame_buffer.append(frame.copy())
            bbox_buffer.append(bbox)

        except FaceNotFoundError:
            cv2.putText(annotated, "No face detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        except MultipleFacesError:
            cv2.putText(annotated, "Multiple faces detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # ── CNN in background every 30 frames ─────────────────────────────
        if frame_count % 30 == 0 and len(frame_buffer) >= 3:
            hud.request_cnn_check(
                classifier,
                list(frame_buffer)[-5:],
                list(bbox_buffer)[-5:],
            )

        # ── Fuse + Render ─────────────────────────────────────────────────
        hud.compute_fusion()
        annotated = hud.render(annotated)

        # FPS
        now = time.perf_counter()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(annotated, f"FPS: {sum(fps_history)/len(fps_history):.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        # ── Trigger Active Challenge ──────────────────────────────────────
        if key == ord(" ") and bbox is not None and hud.face_is_stable:
            log.info("Triggering Flash Challenge...")
            
            # The FlashChallenge takes over the screen and the cap object
            challenge_result = challenger.run(cap)

            if challenge_result["success"]:
                # Send the captured frames to the HSV Glare Trap
                is_live, score, breakdown = analyzer.analyze(
                    challenge_result, bbox=bbox
                )
                hud.flash_result = breakdown["summary"]
                log.info(f"Flash complete. is_live={is_live}, score={score:.3f}")
            else:
                log.warning("Flash capture failed.")
                hud.flash_result = {"is_live": False, "score": 0.0}

            # Force fusion update immediately after flash returns
            hud.compute_fusion()

    cap.release()
    cv2.destroyAllWindows()
    log.info("Phase 3 Demo finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()

    try:
        run_demo(device=args.device, camera_index=args.camera)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()