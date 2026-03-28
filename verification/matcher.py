"""
verification/matcher.py
────────────────────────────────────────────────────────────────────────────
Computes similarity between a probe embedding and a stored gallery embedding.

WHY COSINE SIMILARITY
──────────────────────
Both embeddings are L2-normalised (unit vectors). For unit vectors:

    cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
                            = dot(a, b)          ← because ||a|| = ||b|| = 1

So cosine similarity reduces to a single dot product — extremely fast.

TYPICAL SCORE RANGES (ArcFace buffalo_l)
──────────────────────────────────────────
  Same person, good lighting:      0.65 – 0.85
  Same person, glasses/beard:      0.55 – 0.70
  Different people:                0.10 – 0.40
  Identical twins:                 0.45 – 0.60  ← known hard case

RECOMMENDED THRESHOLD: 0.50
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

from infrastructure.logger import get_logger

log = get_logger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised embeddings.

    Args:
        a: np.ndarray shape (512,), L2-normalised (unit vector).
        b: np.ndarray shape (512,), L2-normalised (unit vector).

    Returns:
        Float in [-1.0, 1.0].
        1.0  = identical vectors (same person, same image)
        0.5+ = likely same person
        0.0  = orthogonal (unrelated)
        <0   = should not happen with real face embeddings

    Note:
        Both inputs MUST be L2-normalised. EmbeddingGenerator.generate()
        and generate_batch() return normalised vectors automatically.
        If you pass unnormalised embeddings the score will be incorrect.
    """
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()

    if a.shape != b.shape:
        raise ValueError(
            f"Embedding shapes must match: got {a.shape} vs {b.shape}"
        )

    # For unit vectors, dot product == cosine similarity
    # We clip to [-1, 1] to guard against floating point drift
    similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))

    log.debug("Cosine similarity: %.4f", similarity)
    return similarity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two embeddings.

    Lower = more similar. For unit vectors, distance is related to
    cosine similarity by: distance = sqrt(2 - 2 * cosine_sim)

    Provided as an alternative metric — cosine similarity is preferred
    for ArcFace embeddings.

    Returns:
        Float ≥ 0.0. Typical range: 0.5–1.2 for same person,
        1.2–2.0 for different people.
    """
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    distance = float(np.linalg.norm(a - b))
    log.debug("Euclidean distance: %.4f", distance)
    return distance