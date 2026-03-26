"""
test_phase3.py — Phase 3 final liveness demo
────────────────────────────────────────────────────────────────────────────
Four liveness signals:
  1. CNN classifier (MobileNetV2) — replaces LBP+SVM
  2. Blink detector (EAR + stability gate)
  3. Optical flow (Farneback)
  4. rPPG pulse detector

Run:
    python test_phase3.py              # CNN + blink + flow (no rPPG yet)
    python test_phase3.py --rppg       # all 4 signals (needs 60+ frames)

Security fix: blink only counts when face is STABLE
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
from liveness.temporal.blink_detector import BlinkDetector, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from liveness.temporal.optical_flow import OpticalFlowChecker
from liveness.temporal.rppg_detector import RPPGDetector
from liveness.spatial.cnn_classifier import CNNClassifier

log = get_logger(__name__)

# ── Weights — blink alone cannot clear threshold ──────────────────────────
W_BLINK = 0.20   # reduced
W_FLOW  = 0.20   # reduced  
W_CNN   = 0.60   # dominant — most reliable
W_RPPG  = 0.00   # disabled until properly implemented
THRESHOLD = 0.40

# Face stability: max bbox centre movement (pixels) to count a blink
STABILITY_MAX_MOVEMENT = 25.0


class LivenessHUD:
    def __init__(self, use_rppg: bool = False):
        self.use_rppg = use_rppg

        # Cumulative blink counter
        self.cumulative_blinks   = 0
        self._prev_window_blinks = 0

        # Scores
        self.cnn_score   = None
        self.flow_score  = 0.0
        self.blink_score = 0.0
        self.rppg_score  = 0.5
        self.fused_score = 0.0
        self.current_ear = 0.0

        # Face stability tracking
        self._bbox_history: deque = deque(maxlen=10)
        self.face_is_stable = True

        # Background CNN thread
        self._cnn_lock    = threading.Lock()
        self._cnn_running = False

    def update_bbox(self, bbox: np.ndarray) -> None:
        """Track face position to detect if it's being moved (phone attack)."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._bbox_history.append((cx, cy))

        if len(self._bbox_history) >= 5:
            positions = np.array(list(self._bbox_history))
            movement  = np.std(positions, axis=0).mean()
            self.face_is_stable = movement < STABILITY_MAX_MOVEMENT
        else:
            self.face_is_stable = True

    def update_blinks(self, window_count: int) -> None:
        """Add delta — only if face was stable during this window."""
        if self.face_is_stable:
            delta = max(0, window_count - self._prev_window_blinks)
            if delta > 0:
                self.cumulative_blinks += delta
        self._prev_window_blinks = window_count

    def request_cnn_check(
        self,
        classifier: CNNClassifier,
        aligned_frames: list[np.ndarray],
    ) -> None:
        """Run CNN check in background thread."""
        with self._cnn_lock:
            if self._cnn_running:
                return
            self._cnn_running = True

        def _run():
            try:
                score = classifier.check(aligned_frames[:5])
                with self._cnn_lock:
                    self.cnn_score = score
            except Exception as e:
                log.warning("CNN check error: %s", e)
            finally:
                with self._cnn_lock:
                    self._cnn_running = False

        threading.Thread(target=_run, daemon=True).start()

    def compute_fused(self) -> float:
        """Weighted fusion of available signals."""
        cnn = self.cnn_score if self.cnn_score is not None else 0.5

        if self.use_rppg:
            total_w = W_BLINK + W_FLOW + W_CNN + W_RPPG
            fused = (
                W_BLINK * self.blink_score +
                W_FLOW  * self.flow_score  +
                W_CNN   * cnn              +
                W_RPPG  * self.rppg_score
            ) / total_w
        else:
            total_w = W_BLINK + W_FLOW + W_CNN
            fused = (
                W_BLINK * self.blink_score +
                W_FLOW  * self.flow_score  +
                W_CNN   * cnn
            ) / total_w

        self.fused_score = float(np.clip(fused, 0.0, 1.0))
        return self.fused_score

    def render(self, frame: np.ndarray) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]

        lines = []
        lines.append(("EAR", f"{self.current_ear:.3f}",
                      self.current_ear >= 0.20))
        lines.append(("Blinks",
                      f"{self.cumulative_blinks}" +
                      (" (stable)" if self.face_is_stable else " (moving!)"),
                      self.cumulative_blinks >= 1))
        lines.append(("Flow",
                      f"{self.flow_score:.3f}",
                      self.flow_score >= 0.3))
        cnn_val = self.cnn_score if self.cnn_score is not None else -1
        lines.append(("CNN",
                      f"{cnn_val:.3f}" if cnn_val >= 0 else "analysing...",
                      cnn_val >= 0.5 if cnn_val >= 0 else True))
        if self.use_rppg:
            lines.append(("rPPG",
                          f"{self.rppg_score:.3f}",
                          self.rppg_score >= 0.5))

        panel_h = len(lines) * 28 + 60
        overlay = out.copy()
        cv2.rectangle(overlay, (10, h - panel_h - 10), (400, h - 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        y = h - panel_h + 4
        for label, value, good in lines:
            color = (180, 180, 180)
            val_color = (0, 210, 0) if good else (0, 100, 255)
            cv2.putText(out, f"{label}:", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(out, value, (120, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color, 1)
            y += 26

        # Fused score bar
        y += 4
        bar_w = int(self.fused_score * 300)
        bar_c = (0, 210, 0) if self.fused_score >= THRESHOLD else (0, 100, 255)
        cv2.putText(out, f"Score: {self.fused_score:.3f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        y += 16
        cv2.rectangle(out, (20, y), (20 + bar_w, y + 10), bar_c, -1)
        cv2.rectangle(out, (20, y), (320, y + 10), (120, 120, 120), 1)
        tx = 20 + int(THRESHOLD * 300)
        cv2.rectangle(out, (tx-1, y-2), (tx+1, y+12), (255,255,255), -1)
        y += 18

        is_live = self.fused_score >= THRESHOLD
        cv2.putText(out, "LIVE" if is_live else "SPOOF",
                    (20, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 210, 0) if is_live else (0, 0, 220), 2)

        # Stability warning
        if not self.face_is_stable:
            cv2.putText(out, "! Face moving — hold still",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        return out


def run_demo(
    use_rppg: bool = False,
    device: str = "cuda",
    camera_index: int = 0,
) -> None:

    log.info("Phase 3 demo — signals: blink+flow+CNN%s",
             "+rPPG" if use_rppg else "")

    # ── Init ──────────────────────────────────────────────────────────────
    detector    = FaceDetector(device=device)
    aligner     = FaceAligner()
    blink_det   = BlinkDetector(ear_threshold=0.20, min_blinks_required=1)
    flow_check  = OpticalFlowChecker(min_frames=8, motion_threshold=0.0005)
    rppg_det    = RPPGDetector() if use_rppg else None
    hud         = LivenessHUD(use_rppg=use_rppg)

    from pathlib import Path
    cnn_path = Path("data/models/mobilenetv2_spoof.pth")
    if not cnn_path.exists():
        log.error(
            "CNN model not found at %s — "
            "run: python scripts/train_cnn_classifier.py",
            cnn_path,
        )
        sys.exit(1)
    classifier = CNNClassifier(model_path=cnn_path, device=device)

    # Buffers
    frame_buffer   = deque(maxlen=90)
    aligned_buffer = deque(maxlen=90)
    bbox_buffer    = deque(maxlen=30)

    fps_history = deque(maxlen=30)
    prev_time   = time.perf_counter()
    frame_count = 0

    blink_det._load_mediapipe()

    window_name = "VeraVisage — Phase 3 Final | Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    log.info("Running — blink to confirm liveness")

    for frame in stream_webcam(camera_index=camera_index):
        frame_count += 1
        annotated = frame.copy()

        # ── Detect + align ────────────────────────────────────────────────
        try:
            face    = detector.detect_single(frame)
            aligned = aligner.align_from_detection(frame, face)
            bbox    = face["bbox"]

            cv2.rectangle(annotated,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 220, 0), 2)

            frame_buffer.append(frame.copy())
            aligned_buffer.append(aligned.copy())
            bbox_buffer.append(bbox.tolist())
            hud.update_bbox(bbox)

        except FaceNotFoundError:
            cv2.putText(annotated, "No face — look at camera",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        except MultipleFacesError:
            cv2.putText(annotated, "Multiple faces — one person only",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        except Exception as e:
            log.warning("Detection: %s", e)
            continue

        # ── Per-frame EAR ─────────────────────────────────────────────────
        lm = blink_det._get_landmarks_from_frame(frame)
        if lm is not None:
            l = blink_det._compute_ear(lm, LEFT_EYE_INDICES)
            r = blink_det._compute_ear(lm, RIGHT_EYE_INDICES)
            hud.current_ear = (l + r) / 2.0

        # ── Blink recount every 15 frames ─────────────────────────────────
        if frame_count % 15 == 0 and len(frame_buffer) >= 5:
            wc, _ = blink_det.count_blinks(list(frame_buffer))
            hud.update_blinks(wc)
            hud.blink_score = 1.0 if hud.cumulative_blinks >= 1 else 0.0

        # ── Optical flow every 20 frames ──────────────────────────────────
        if frame_count % 20 == 0 and len(frame_buffer) >= 8:
            hud.flow_score = flow_check.check(list(frame_buffer))

        # ── CNN in background every 60 frames ─────────────────────────────
        if frame_count % 60 == 0 and len(aligned_buffer) >= 3:
            hud.request_cnn_check(classifier, list(aligned_buffer))

        # ── rPPG every 90 frames (needs 60+ frames) ───────────────────────
        if use_rppg and rppg_det and frame_count % 90 == 0 and \
                len(frame_buffer) >= 60:
            bboxes = list(bbox_buffer)
            frames_list = list(frame_buffer)
            hud.rppg_score = rppg_det.check(frames_list, bboxes)

        # ── Fuse + render ─────────────────────────────────────────────────
        hud.compute_fused()
        annotated = hud.render(annotated)

        # FPS
        now = time.perf_counter()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(annotated,
                    f"FPS: {sum(fps_history)/len(fps_history):.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 1)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    log.info("Done — blinks=%d", hud.cumulative_blinks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rppg",   action="store_true",
                        help="Enable rPPG pulse detection (needs 2+ sec)")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()

    try:
        run_demo(
            use_rppg=args.rppg,
            device=args.device,
            camera_index=args.camera,
        )
    except KeyboardInterrupt:
        cv2.destroyAllWindows()