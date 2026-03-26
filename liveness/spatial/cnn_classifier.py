"""
liveness/spatial/cnn_classifier.py
────────────────────────────────────────────────────────────────────────────
CNN-based liveness detection using fine-tuned MobileNetV2.

Replaces texture_analyzer.py (LBP + SVM) with a deep learning classifier
that runs on GPU and generalises across camera domains.

ARCHITECTURE
─────────────
MobileNetV2 (ImageNet pretrained)
  → Frozen backbone (efficient feature extraction)
  → Binary classifier head (live=1, spoof=0)
  → Trained on LCC-FASD dataset
  → ~2ms inference per frame on RTX 4050

WHY THIS REPLACES LBP+SVM
───────────────────────────
LBP+SVM learns handcrafted texture patterns that are camera-specific.
MobileNetV2 learns deep hierarchical features that generalise across
cameras, lighting conditions, and subjects. Published results on LCC-FASD
show 94-98% accuracy vs ~84% for LBP+SVM.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from infrastructure.exceptions import DeepfakeAuthError
from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CNNClassifier(BaseLivenessCheck):
    """
    MobileNetV2-based live/spoof classifier.
    Implements BaseLivenessCheck — plugs directly into LivenessAggregator.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        self.device     = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._model     = None
        self.model_path = Path(model_path) if model_path else None

        if self.model_path and self.model_path.exists():
            self.load_model(self.model_path)
        elif self.model_path:
            log.warning(
                "CNNClassifier: model not found at %s — "
                "run scripts/train_cnn_classifier.py first.",
                self.model_path,
            )

        log.info("CNNClassifier initialised — device=%s", self.device)

    @property
    def name(self) -> str:
        return "cnn_spoof"

    def _build_model(self) -> nn.Module:
        """Build MobileNetV2 with binary classification head."""
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model

    def load_model(self, path: str | Path) -> None:
        """Load trained weights from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CNN model not found at {path}. "
                f"Run: python scripts/train_cnn_classifier.py"
            )

        model = self._build_model()
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)
        self._model = model

        dev_acc = checkpoint.get("dev_acc", 0.0)
        log.info(
            "CNNClassifier loaded — dev_acc=%.4f, device=%s",
            dev_acc, self.device,
        )

    def predict_single(self, image: np.ndarray) -> tuple[int, float]:
        """
        Run inference on one BGR image.

        Returns:
            (predicted_class, live_probability)
            predicted_class: 1=live, 0=spoof
        """
        if self._model is None:
            raise DeepfakeAuthError("No model loaded — call load_model() first.")

        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = INFERENCE_TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)
            live_prob     = float(probs[0, 1].item())
            predicted_cls = int(probs[0].argmax().item())

        return predicted_cls, live_prob

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run CNN liveness check — averages over up to 5 frames.

        Args:
            frames: BGR aligned face crops from FaceAligner.

        Returns:
            Live probability in [0.0, 1.0].
        """
        if self._model is None:
            raise DeepfakeAuthError(
                "CNNClassifier: no model loaded. "
                "Run scripts/train_cnn_classifier.py first."
            )

        if not frames:
            log.warning("CNNClassifier.check() received empty frames")
            return 0.0

        scores = []
        for frame in frames[:5]:
            _, live_prob = self.predict_single(frame)
            scores.append(live_prob)
            log.debug("CNN frame score: %.4f", live_prob)

        final_score = float(np.mean(scores))
        log.info("CNNClassifier score: %.4f (avg %d frames)",
                 final_score, len(scores))
        return final_score