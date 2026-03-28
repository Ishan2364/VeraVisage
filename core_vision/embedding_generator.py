"""
core_vision/embedding_generator.py
────────────────────────────────────────────────────────────────────────────
Generates 512-dimensional L2-normalised face embeddings using ArcFace
via the insightface library (buffalo_l model).

WHY ARCFACE
────────────
ArcFace (Additive Angular Margin Loss) is the gold standard for face
recognition. It trains the embedding space so that:
  - Same person embeddings cluster tightly together
  - Different person embeddings are pushed far apart
  - The margin is enforced in angular space, not Euclidean

Result: cosine similarity between two embeddings of the same person
reliably exceeds 0.6, while different people score below 0.4.

MODEL: buffalo_l
─────────────────
insightface's buffalo_l pack includes:
  - RetinaFace detector (we already use this in face_detector.py)
  - ArcFace recognition model (what we use here)
  - 5-point landmark model

We load ONLY the recognition module to avoid loading a second detector.
The aligned 112×112 crop from face_aligner.py is fed directly.

OUTPUT
──────
  np.ndarray shape (512,), dtype float32, L2-normalised (unit vector)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer, l2_normalize

log = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates L2-normalised 512-d ArcFace embeddings from aligned face crops.

    Input:  112×112 BGR aligned crop from FaceAligner
    Output: np.ndarray shape (512,), unit norm, float32
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: "cuda" for GPU inference, "cpu" for fallback.
        """
        self.device  = device
        self._model  = None

        log.info("EmbeddingGenerator initialised — device=%s", device)

    def _load_model(self) -> None:
        """Lazy-load ArcFace on first use."""
        if self._model is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is not installed.\n"
                "Run: pip install insightface onnxruntime-gpu"
            )

        log.info(
            "Loading ArcFace (buffalo_l) — "
            "first run downloads weights (~300 MB)..."
        )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        # We MUST allow all modules (or at least detection and recognition)
        # because app.prepare() asserts that 'detection' is in self.models
        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        app.prepare(ctx_id=0 if self.device == "cuda" else -1)

        # Extract just the recognition model
        # FaceAnalysis models is a dictionary in some versions, list in others
        self._model = None
        if hasattr(app, "models") and isinstance(app.models, dict):
            self._model = app.models.get("recognition")
        else:
            # Fallback: try getting it by iterating models list/dict
            items = getattr(app, "models", [])
            items_to_iter = items.items() if isinstance(items, dict) else enumerate(items)
            for _, model in items_to_iter:
                # Sometimes models don't have taskname, use __class__.__name__ or taskname
                task = getattr(model, "taskname", getattr(model.__class__, "__name__", ""))
                if "rec" in task.lower() or "recognition" in task.lower():
                    self._model = model
                    break

        if self._model is None:
            raise RuntimeError(
                "Could not load ArcFace recognition model from buffalo_l. "
                "Try: pip install --upgrade insightface"
            )

        log.info("ArcFace recognition model loaded on %s", self.device)

    @timer
    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate a 512-d embedding from one aligned face crop.

        Args:
            aligned_face: BGR np.ndarray, shape (112, 112, 3), dtype uint8.
                          Must come from FaceAligner — ArcFace was trained
                          on this exact canonical alignment.

        Returns:
            np.ndarray shape (512,), dtype float32, L2-normalised.
            Cosine similarity between this and another embedding is
            directly comparable — no further normalisation needed.

        Raises:
            ValueError: If input shape is wrong.
            RuntimeError: If the model fails to produce an embedding.
        """
        self._load_model()

        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("aligned_face is empty or None")

        if aligned_face.shape != (112, 112, 3):
            raise ValueError(
                f"aligned_face must be shape (112, 112, 3), "
                f"got {aligned_face.shape}. "
                f"Ensure FaceAligner was used before calling generate()."
            )

        # insightface recognition models expect BGR uint8 (same as OpenCV)
        embedding = self._model.get_feat(aligned_face)

        if embedding is None or len(embedding) == 0:
            raise RuntimeError(
                "ArcFace returned empty embedding. "
                "Check that the aligned crop is valid."
            )

        # Flatten in case model returns (1, 512)
        embedding = np.array(embedding).flatten().astype(np.float32)

        # L2 normalise — ensures cosine similarity == dot product
        embedding = l2_normalize(embedding)

        log.debug(
            "Embedding generated — shape=%s, norm=%.4f",
            embedding.shape, np.linalg.norm(embedding),
        )
        return embedding

    def generate_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of aligned crops and average them.

        This is the enrolment method — 30 frames averaged into one
        robust stored embedding that handles natural pose micro-variation.

        Args:
            aligned_faces: List of (112, 112, 3) BGR crops.

        Returns:
            Single L2-normalised embedding shape (512,) — the average
            of all individual embeddings.
        """
        if not aligned_faces:
            raise ValueError("aligned_faces list is empty")

        embeddings = []
        for i, face in enumerate(aligned_faces):
            try:
                emb = self.generate(face)
                embeddings.append(emb)
            except Exception as e:
                log.warning("Skipping frame %d — embedding failed: %s", i, repr(e))

        if not embeddings:
            raise RuntimeError(
                "All frames failed embedding generation. "
                "Check face alignment quality."
            )

        # Stack and average
        stacked = np.vstack(embeddings)          # (N, 512)
        averaged = stacked.mean(axis=0)           # (512,)
        averaged = l2_normalize(averaged)          # unit norm

        log.info(
            "Batch embedding — %d/%d frames successful, "
            "final norm=%.4f",
            len(embeddings), len(aligned_faces),
            np.linalg.norm(averaged),
        )
        return averaged