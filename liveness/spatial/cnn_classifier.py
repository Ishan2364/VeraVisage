"""
liveness/spatial/cnn_classifier.py
────────────────────────────────────────────────────────────────────────────
CNN-based liveness detection using MiniFASNet via uniface library.

Replaces the domain-limited MobileNetV2 trained on LCC-FASD with
MiniFASNet — a pretrained model that generalises across cameras with
no additional training required.

WHY MiniFASNet OVER MobileNetV2
────────────────────────────────
MobileNetV2 trained on LCC-FASD scored 0.019 on a real face from your
webcam because LCC-FASD images are YouTube-quality frames — a different
domain from your Acer laptop webcam.

MiniFASNet was trained on VGGFace2 (3.3M faces, 9,000 subjects) with
Fourier spectrum supervision — it learns frequency-domain spoof artifacts
that are camera-independent. Result: works on any webcam out of the box.

MODELS (auto-downloaded on first use, ~1.2 MB each)
─────────────────────────────────────────────────────
  MiniFASNetV2   (default, recommended) — scale factor 2.7
  MiniFASNetV1SE (alternative)          — scale factor 4.0

USAGE
─────
    classifier = CNNClassifier(device="cuda")
    score = classifier.check([aligned_frame1, aligned_frame2, ...])
    # score in [0.0, 1.0] — 1.0 = live, 0.0 = spoof
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.exceptions import DeepfakeAuthError
from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)


class CNNClassifier(BaseLivenessCheck):
    """
    MiniFASNet-based live/spoof classifier via uniface library.

    Implements BaseLivenessCheck — drop-in replacement for the old
    MobileNetV2-based classifier. Same interface, better generalisation.

    Models auto-download to ~/.uniface/models/ on first use (~1.2 MB).
    """

    def __init__(
        self,
        device: str = "cuda",
        model_variant: str = "v2",
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            device:               "cuda" for GPU, "cpu" for fallback.
                                  uniface uses onnxruntime-gpu if available.
            model_variant:        "v2" (default, recommended) or "v1se".
            confidence_threshold: Score above this = live. Used for logging
                                  only — check() returns raw probability.
        """
        self.device               = device
        self.model_variant        = model_variant
        self.confidence_threshold = confidence_threshold
        self._spoofer             = None
        self._detector            = None

        log.info(
            "CNNClassifier (MiniFASNet-%s) initialised — device=%s",
            model_variant.upper(), device,
        )

    @property
    def name(self) -> str:
        return "cnn_spoof"

    def _load_model(self) -> None:
        """Lazy-load MiniFASNet on first use. Downloads weights if needed."""
        if self._spoofer is not None:
            return

        try:
            from uniface.spoofing import MiniFASNet
            from uniface.constants import MiniFASNetWeights
        except ImportError:
            raise ImportError(
                "uniface is not installed. Run:\n"
                "  pip install uniface\n"
                "  pip install uniface[gpu]  # for GPU support"
            )

        # Select model variant
        if self.model_variant == "v1se":
            model_name = MiniFASNetWeights.V1SE
        else:
            model_name = MiniFASNetWeights.V2  # default

        # Set ONNX execution provider based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self._spoofer = MiniFASNet(
                model_name=model_name,
                providers=providers,
            )
            log.info(
                "MiniFASNet-%s loaded — weights auto-downloaded if first run",
                self.model_variant.upper(),
            )
        except TypeError:
            # Some versions don't accept providers parameter
            self._spoofer = MiniFASNet(model_name=model_name)
            log.info("MiniFASNet loaded (no provider param)")

    def predict_single(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | list | np.ndarray,
    ) -> float:
        """
        Run MiniFASNet on a single frame with a bounding box.

        Args:
            frame: Full BGR camera frame (NOT the aligned crop).
                   MiniFASNet needs the full frame + bbox to crop internally
                   at its own scale factor.
            bbox:  Face bounding box [x1, y1, x2, y2].

        Returns:
            Live probability in [0.0, 1.0].
        """
        self._load_model()

        result = self._spoofer.predict(frame, bbox)

        # result.is_real: True = live, False = spoof
        # result.confidence: probability of the predicted class
        if result.is_real:
            live_score = float(result.confidence)
        else:
            # Spoof predicted — live score is 1 - confidence
            live_score = float(1.0 - result.confidence)

        log.debug(
            "MiniFASNet: is_real=%s confidence=%.3f → live_score=%.3f",
            result.is_real, result.confidence, live_score,
        )
        return live_score

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run CNN liveness check — averages over up to 5 frames.

        NOTE: This method accepts aligned 112×112 crops from FaceAligner
        but MiniFASNet works better on full frames with bbox. Since we
        only have crops here, we pass them directly — MiniFASNet uses
        the full image as both frame and assumes bbox covers the whole image.

        For best accuracy, use check_with_bbox() if you have full frames.

        Args:
            frames: BGR aligned face crops from FaceAligner (112×112).

        Returns:
            Live probability in [0.0, 1.0].
        """
        self._load_model()

        if not frames:
            log.warning("CNNClassifier.check() received empty frames list")
            return 0.0

        scores = []
        for frame in frames[:5]:
            try:
                h, w = frame.shape[:2]
                bbox = [0, 0, w, h]  # full image as bbox for aligned crops
                score = self.predict_single(frame, bbox)
                scores.append(score)
            except Exception as e:
                log.warning("MiniFASNet inference error: %s", e)

        if not scores:
            return 0.0

        final_score = float(np.mean(scores))
        log.info(
            "CNNClassifier score: %.4f (avg %d frames)",
            final_score, len(scores),
        )
        return final_score

    def check_with_bbox(
        self,
        full_frames: list[np.ndarray],
        bboxes: list,
    ) -> float:
        """
        Run MiniFASNet with full frames and bounding boxes.

        This gives better accuracy than check() because MiniFASNet
        can use its internal scale factor to crop at the right resolution.

        Args:
            full_frames: List of full BGR camera frames.
            bboxes:      List of [x1, y1, x2, y2] bounding boxes,
                         one per frame.

        Returns:
            Live probability in [0.0, 1.0].
        """
        self._load_model()

        if not full_frames or not bboxes:
            return 0.0

        scores = []
        pairs  = list(zip(full_frames, bboxes))[:5]

        for frame, bbox in pairs:
            try:
                score = self.predict_single(frame, bbox)
                scores.append(score)
            except Exception as e:
                log.warning("MiniFASNet bbox inference error: %s", e)

        if not scores:
            return 0.0

        final_score = float(np.mean(scores))
        log.info(
            "CNNClassifier (bbox mode) score: %.4f (avg %d frames)",
            final_score, len(scores),
        )
        return final_score