"""
core_vision/face_detector.py
────────────────────────────────────────────────────────────────────────────
Wraps insightface's RetinaFace detector to return structured detection
results (bounding box + 5-point landmarks + confidence) for each face
found in a frame.

WHY RETINAFACE
──────────────
RetinaFace is a single-shot multi-task network that detects faces AND
regresses 5 facial landmarks simultaneously. This matters because:
  - The landmarks are used directly by face_aligner.py for alignment
  - No second model pass is needed (unlike some older pipelines)
  - It handles scale variation well — works from close-up to ~5m away
  - insightface's implementation runs on ONNX, so CPU/GPU is just a
    flag — no code change needed between machines

OUTPUT FORMAT
─────────────
Each detected face is returned as a dict:
{
    "bbox":       np.ndarray shape (4,)   — [x1, y1, x2, y2] pixels
    "landmarks":  np.ndarray shape (5, 2) — [[x,y], ...] pixel coords
                  order: left_eye, right_eye, nose, left_mouth, right_mouth
    "confidence": float                   — detection score in [0, 1]
}

This dict schema is the contract between face_detector.py and every
module that consumes it (face_aligner.py, the pipeline, tests).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.exceptions import FaceNotFoundError, MultipleFacesError
from infrastructure.logger import get_logger
from infrastructure.utils import timer

log = get_logger(__name__)


class FaceDetector:
    """
    Detects faces in BGR frames using insightface's RetinaFace backend.

    Usage:
        detector = FaceDetector(device="cuda")  # or "cpu"
        results  = detector.detect(frame)        # list of face dicts
        face     = detector.detect_single(frame) # exactly one face or raises
    """

    def __init__(
        self,
        device: str = "cuda",
        detection_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            device:               "cuda" uses your NVIDIA GPU via onnxruntime-gpu.
                                  "cpu"  falls back to CPU (slower but always works).
            detection_size:       Internal resolution the model resizes frames to.
                                  (640, 640) is the standard — don't change unless
                                  you need to trade accuracy for speed on very slow hardware.
            confidence_threshold: Detections below this score are discarded.
                                  0.5 is conservative — raise to 0.7 to reduce false
                                  positives in crowded scenes.
        """
        self.device = device
        self.detection_size = detection_size
        self.confidence_threshold = confidence_threshold
        self._model = None  # lazy-loaded on first detect() call

        log.info(
            "FaceDetector initialised — backend=RetinaFace, device=%s, "
            "det_size=%s, conf_threshold=%.2f",
            device, detection_size, confidence_threshold,
        )

    def _load_model(self) -> None:
        """
        Lazy-load the insightface FaceAnalysis model.

        We load on first use rather than in __init__ so that importing
        this module doesn't trigger a heavy model download/load at import
        time — only when you actually call detect().

        insightface will download the RetinaFace weights (~1.7 MB) to
        ~/.insightface/models/ on first run. Subsequent runs use the cache.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is not installed. Run:\n"
                "  pip install insightface onnxruntime-gpu\n"
                "(use onnxruntime instead of onnxruntime-gpu if you have no GPU)"
            )

        log.info("Loading RetinaFace model — first run will download weights (~1.7 MB)...")

        # providers controls whether onnxruntime uses GPU or CPU
        # CUDAExecutionProvider → onnxruntime-gpu uses your NVIDIA card
        # CPUExecutionProvider  → fallback, always available
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # FaceAnalysis bundles detection + landmark models together.
        # allowed_modules=["detection"] means we ONLY load the detector here —
        # we do NOT load the recognition/embedding model (that lives in
        # embedding_generator.py). This keeps memory usage low.
        self._model = FaceAnalysis(
            name="buffalo_sc",           # lightweight model: RetinaFace det + scrfd
            allowed_modules=["detection"],
            providers=providers,
        )

        self._model.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=self.detection_size,
        )

        log.info("RetinaFace model loaded successfully on %s", self.device)

    @timer
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame: np.ndarray shape (H, W, 3), dtype uint8, BGR colour order.
                   This is OpenCV's native format — no conversion needed.

        Returns:
            List of face dicts (may be empty if no faces found):
            [
                {
                    "bbox":       np.ndarray([x1, y1, x2, y2]),
                    "landmarks":  np.ndarray shape (5, 2),
                    "confidence": float,
                },
                ...
            ]
            Sorted by confidence descending — highest-confidence face first.
        """
        self._load_model()

        if frame is None or frame.size == 0:
            raise ValueError("detect() received an empty or None frame.")

        # insightface expects BGR (same as OpenCV) — no conversion needed
        raw_faces = self._model.get(frame)

        if not raw_faces:
            log.debug("No faces detected in frame shape=%s", frame.shape)
            return []

        results = []
        for face in raw_faces:
            confidence = float(face.det_score)

            if confidence < self.confidence_threshold:
                log.debug("Face discarded — confidence %.3f < threshold %.3f",
                          confidence, self.confidence_threshold)
                continue
            kps = getattr(face, "kps", None)
            if kps is None:
                kps = getattr(face, "landmark_2d_106", None)
            if kps is None:
                kps = np.zeros((5, 2), dtype=np.float32)    
            results.append({
                "bbox":       face.bbox.astype(np.float32),
                "landmarks":  kps.astype(np.float32),
                "confidence": confidence,
            })

        # Sort by confidence — best detection first
        results.sort(key=lambda f: f["confidence"], reverse=True)

        log.debug("Detected %d face(s) above threshold in frame", len(results))
        return results

    def detect_single(
        self,
        frame: np.ndarray,
        allow_multiple: bool = False,
    ) -> dict:
        """
        Detect exactly one face. Raises if zero or multiple faces are found.

        This is what the authentication pipeline calls — it expects exactly
        one person in front of the camera.

        Args:
            frame:           BGR frame, same as detect().
            allow_multiple:  If True, returns the highest-confidence face
                             even when multiple are detected (instead of
                             raising MultipleFacesError). Useful during
                             enrolment where we can tell the user to step
                             closer rather than failing hard.

        Returns:
            Single face dict with "bbox", "landmarks", "confidence".

        Raises:
            FaceNotFoundError:  If no face is detected.
            MultipleFacesError: If >1 face detected and allow_multiple=False.
        """
        faces = self.detect(frame)

        if len(faces) == 0:
            raise FaceNotFoundError(
                "No face detected in this frame. "
                "Ensure your face is clearly visible and well-lit."
            )

        if len(faces) > 1 and not allow_multiple:
            raise MultipleFacesError(
                f"{len(faces)} faces detected. "
                "Only one person should be in front of the camera during authentication."
            )

        # Return highest-confidence face (already sorted by detect())
        return faces[0]