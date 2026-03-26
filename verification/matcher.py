"""
matcher.py
──────────
Cosine / Euclidean similarity computation between probe and gallery embeddings.
"""
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine similarity in [-1, 1] between two unit vectors."""
    raise NotImplementedError


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Returns L2 distance between two embedding vectors."""
    raise NotImplementedError
