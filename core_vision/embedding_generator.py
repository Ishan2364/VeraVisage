"""
embedding_generator.py
──────────────────────
Runs the backbone (ArcFace / FaceNet) to produce a 512-d unit embedding.
"""
import numpy as np


class EmbeddingGenerator:
    """Converts an aligned face crop into a fixed-length identity vector."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        """Returns an L2-normalised embedding of shape (512,)."""
        raise NotImplementedError
