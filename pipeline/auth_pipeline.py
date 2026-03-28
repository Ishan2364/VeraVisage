"""
pipeline/auth_pipeline.py
────────────────────────────────────────────────────────────────────────────
End-to-end authentication pipeline integrating Phase 3 (Liveness) and
Phase 4 (Face Verification).

AUTHENTICATION FLOW
────────────────────
1. Open webcam, detect face
2. Run CNN liveness gate (MiniFASNet) continuously
3. User presses SPACE → Flash challenge runs (HSV dual-veto)
4. If dual-veto LIVE:
   a. Generate ArcFace embedding from recent frames
   b. Retrieve stored embedding for claimed user_id
   c. Compute cosine similarity
   d. DecisionEngine → ACCEPT or REJECT
5. Return structured verdict

INTEGRATION WITH PHASE 3
──────────────────────────
Face matching (Phase 4) ONLY runs if liveness passes.
This is the core security property:
  - A deepfake that passes CNN is stopped by flash challenge
  - A flash challenge that passes is stopped by identity mismatch
  - All three gates must clear for ACCEPT
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import threading
from collections import deque

import cv2
import numpy as np

from core_vision.face_detector import FaceDetector
from core_vision.face_aligner import FaceAligner
from core_vision.embedding_generator import EmbeddingGenerator
from infrastructure.exceptions import (
    FaceNotFoundError, MultipleFacesError, IdentityNotFoundError,
)
from infrastructure.logger import get_logger
from liveness.active.flash_challenge import FlashChallenge
from liveness.active.reflection_analyzer import ReflectionAnalyzer
from liveness.spatial.cnn_classifier import CNNClassifier
from verification.identity_store import IdentityStore
from verification.matcher import cosine_similarity
from verification.decision_engine import DecisionEngine

log = get_logger(__name__)

CNN_VETO_GATE   = 0.25
FLASH_VETO_GATE = 0.50
THRESHOLD       = 0.45


def run_authentication(
    user_id: str,
    device: str = "cuda",
    camera_index: int = 0,
    similarity_threshold: float = 0.50,
) -> dict:
    """
    Full authentication: liveness + face verification.

    Args:
        user_id:              The identity the person claims to be.
        device:               "cuda" or "cpu".
        camera_index:         Webcam index.
        similarity_threshold: Cosine similarity threshold for identity match.

    Returns:
        Verdict dict:
        {
            "accepted":         bool,
            "user_id":          str,
            "similarity":       float,
            "liveness_passed":  bool,
            "liveness_score":   float,
            "reason":           str,
        }
    """
    log.info("Authentication started for user '%s'", user_id)

    # ── Init ──────────────────────────────────────────────────────────────
    detector   = FaceDetector(device=device)
    aligner    = FaceAligner()
    embedder   = EmbeddingGenerator(device=device)
    cnn        = CNNClassifier(device=device)
    challenger = FlashChallenge(camera_index=camera_index, sequence_length=4)
    analyzer   = ReflectionAnalyzer(
        min_alignment=0.30, min_colours_passing=3
    )
    store      = IdentityStore()
    engine     = DecisionEngine(similarity_threshold=similarity_threshold)

    # Check user is enrolled before even opening camera
    if not store.is_enrolled(user_id):
        return {
            "accepted":        False,
            "user_id":         user_id,
            "similarity":      0.0,
            "liveness_passed": False,
            "liveness_score":  0.0,
            "reason": (
                f"User '{user_id}' is not enrolled. "
                f"Run enrolment first."
            ),
        }

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "accepted":        False,
            "user_id":         user_id,
            "similarity":      0.0,
            "liveness_passed": False,
            "liveness_score":  0.0,
            "reason":          f"Cannot open camera {camera_index}",
        }

    for _ in range(10):
        cap.read()

    # Shared state
    frame_buffer = deque(maxlen=30)
    bbox_buffer  = deque(maxlen=30)
    aligned_buffer = deque(maxlen=30)

    cnn_score     = [0.5]   # mutable for thread access
    cnn_lock      = threading.Lock()
    cnn_running   = [False]
    flash_result  = [None]
    frame_count   = [0]
    face_bbox     = [None]

    fps_history = deque(maxlen=30)
    prev_time   = time.perf_counter()

    verdict = None

    window_name = f"VeraVisage — Auth: {user_id} | SPACE=Flash, Q=cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    def request_cnn():
        with cnn_lock:
            if cnn_running[0] or not frame_buffer:
                return
            cnn_running[0] = True
            frames = list(frame_buffer)[-5:]
            bboxes = list(bbox_buffer)[-5:]

        def _run():
            try:
                score = cnn.check_with_bbox(frames, bboxes)
                with cnn_lock:
                    cnn_score[0] = score
            except Exception as e:
                log.warning("CNN error: %s", e)
            finally:
                with cnn_lock:
                    cnn_running[0] = False

        threading.Thread(target=_run, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count[0] += 1
            annotated = frame.copy()

            # ── Detect ────────────────────────────────────────────────────
            bbox = None
            try:
                face    = detector.detect_single(frame)
                bbox    = face["bbox"].tolist()
                aligned = aligner.align_from_detection(frame, face)
                face_bbox[0] = bbox

                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              (0, 210, 0), 2)

                frame_buffer.append(frame.copy())
                bbox_buffer.append(bbox)
                aligned_buffer.append(aligned.copy())

            except FaceNotFoundError:
                cv2.putText(annotated, "No face — look at camera",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 100, 255), 2)
            except MultipleFacesError:
                cv2.putText(annotated, "Multiple faces — one person only",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 100, 255), 2)

            # CNN every 30 frames
            if frame_count[0] % 30 == 0:
                request_cnn()

            # ── HUD ───────────────────────────────────────────────────────
            h_frame = annotated.shape[0]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, h_frame - 170),
                          (420, h_frame - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            y = h_frame - 150

            # CNN row
            c_score = cnn_score[0]
            cnn_col = (0, 210, 0) if c_score >= CNN_VETO_GATE \
                      else (0, 0, 210)
            veto_tag = " [VETO]" if c_score < CNN_VETO_GATE else " [PASS]"
            cv2.putText(annotated, f"CNN:   {c_score:.3f}{veto_tag}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, cnn_col, 1)
            y += 28

            # Flash row
            if flash_result[0] is None:
                cv2.putText(annotated, "Flash: waiting — press SPACE",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150, 150, 150), 1)
            else:
                f_score = flash_result[0].get("score", 0.0)
                f_live  = flash_result[0].get("is_live", False)
                f_col   = (0, 210, 0) if f_live else (0, 0, 210)
                f_tag   = "[PASS]" if f_live else "[VETO]"
                cv2.putText(annotated,
                            f"Flash: {f_score:.3f} {f_tag}",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, f_col, 1)
            y += 28

            # Identity row
            cv2.putText(annotated, f"User:  {user_id}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1)
            y += 28

            # Verdict row
            if verdict is not None:
                v_col  = (0, 210, 0) if verdict["accepted"] else (0, 0, 210)
                v_text = "ACCEPT" if verdict["accepted"] else "REJECT"
                cv2.putText(annotated, f"{v_text}  sim={verdict['similarity']:.3f}",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, v_col, 2)
            else:
                cv2.putText(annotated, "AWAITING FLASH",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 200, 255), 2)

            # FPS
            now = time.perf_counter()
            fps_history.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(annotated,
                        f"FPS: {sum(fps_history)/len(fps_history):.1f}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return {
                    "accepted":        False,
                    "user_id":         user_id,
                    "similarity":      0.0,
                    "liveness_passed": False,
                    "liveness_score":  0.0,
                    "reason":          "Cancelled by user",
                }

            # ── SPACE: run flash challenge ─────────────────────────────────
            if key == ord(" ") and bbox is not None:
                log.info("Flash challenge triggered")

                challenge_result = challenger.run(cap)

                if challenge_result["success"]:
                    is_live, f_score, breakdown = analyzer.analyze(
                        challenge_result, bbox=bbox
                    )
                    flash_result[0] = breakdown["summary"]
                    flash_result[0]["score"]   = f_score
                    flash_result[0]["is_live"] = is_live
                else:
                    flash_result[0] = {"is_live": False, "score": 0.0}
                    is_live  = False
                    f_score  = 0.0

                # ── Dual-veto fusion ───────────────────────────────────────
                c_score    = cnn_score[0]
                liveness_ok = (
                    c_score >= CNN_VETO_GATE and
                    is_live and
                    f_score >= FLASH_VETO_GATE
                )
                fused_liveness = float(
                    np.clip(0.40 * c_score + 0.60 * f_score, 0.0, 1.0)
                ) if liveness_ok else 0.0

                if not liveness_ok:
                    verdict = engine.decide(
                        similarity=0.0,
                        liveness_passed=False,
                        liveness_score=fused_liveness,
                        user_id=user_id,
                    )
                    log.info("Liveness failed — skipping face matching")
                    continue

                # ── Face matching (only if liveness passed) ────────────────
                log.info(
                    "Liveness PASSED (cnn=%.3f, flash=%.3f) — "
                    "running face matching",
                    c_score, f_score,
                )

                try:
                    # Generate embedding from recent aligned frames
                    recent_aligned = list(aligned_buffer)[-20:]
                    if len(recent_aligned) < 5:
                        raise ValueError("Not enough aligned frames")

                    probe_embedding   = embedder.generate_batch(recent_aligned)
                    gallery_embedding = store.retrieve(user_id)
                    similarity        = cosine_similarity(
                        probe_embedding, gallery_embedding
                    )

                    verdict = engine.decide(
                        similarity=similarity,
                        liveness_passed=True,
                        liveness_score=fused_liveness,
                        user_id=user_id,
                    )

                except IdentityNotFoundError:
                    verdict = {
                        "accepted":        False,
                        "user_id":         user_id,
                        "similarity":      0.0,
                        "liveness_passed": True,
                        "liveness_score":  fused_liveness,
                        "reason":          f"User '{user_id}' not found in store.",
                    }
                except Exception as e:
                    log.error("Face matching error: %s", e)
                    verdict = {
                        "accepted":        False,
                        "user_id":         user_id,
                        "similarity":      0.0,
                        "liveness_passed": True,
                        "liveness_score":  fused_liveness,
                        "reason":          f"Matching error: {e}",
                    }

                log.info("Auth verdict: %s", verdict)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if verdict is None:
        return {
            "accepted":        False,
            "user_id":         user_id,
            "similarity":      0.0,
            "liveness_passed": False,
            "liveness_score":  0.0,
            "reason":          "Authentication did not complete.",
        }

    return verdict