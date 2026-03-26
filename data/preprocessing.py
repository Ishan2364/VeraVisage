"""
preprocessing.py
────────────────
Resize, normalize, and convert raw frames before they enter any model.
"""
import numpy as np


def normalize_face(image: np.ndarray) -> np.ndarray:
    """Scales pixel values to [0, 1] and applies ImageNet mean/std."""
    raise NotImplementedError


def resize_face(image: np.ndarray, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Resizes a face crop to the target size using INTER_LINEAR."""
    raise NotImplementedError
