"""
infrastructure/utils.py
────────────────────────────────────────────────────────────────────────────
Pure utility functions used across the entire project.

WHAT BELONGS HERE
─────────────────
  - Functions with no project-specific imports (no models, no config)
  - Helpers used by 3+ different modules
  - Things that are genuinely "utilities": I/O, timing, formatting

WHAT DOES NOT BELONG HERE
──────────────────────────
  - Business logic (goes in the relevant domain module)
  - Config loading (goes in config_loader.py)
  - Logging setup (goes in logger.py)
  - Anything that imports from core_vision, liveness, etc.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np

from infrastructure.logger import get_logger

log = get_logger(__name__)

# TypeVar so the @timer decorator preserves the wrapped function's type hints
F = TypeVar("F", bound=Callable[..., Any])


# ══════════════════════════════════════════════════════════════════════════
# Image I/O
# ══════════════════════════════════════════════════════════════════════════

def read_image(path: str | Path) -> np.ndarray:
    """
    Read an image from disk and return it as a BGR NumPy array.

    OpenCV's default is BGR (not RGB). Every other module in this project
    works in BGR to stay consistent with OpenCV. If you need RGB, call
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB) at the point of use.

    Args:
        path: Absolute or relative path to the image file.
              Supports JPEG, PNG, BMP, TIFF, and most common formats.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, BGR colour order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If OpenCV cannot decode the file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"OpenCV could not read image at {path}. "
            f"File may be corrupt or an unsupported format."
        )

    log.debug("read_image: %s  shape=%s", path.name, image.shape)
    return image


def write_image(image: np.ndarray, path: str | Path) -> None:
    """
    Write a BGR NumPy array to disk as an image file.

    The output format is inferred from the file extension
    (e.g. ".jpg" → JPEG, ".png" → PNG).

    Args:
        image: np.ndarray of shape (H, W, 3) or (H, W), dtype uint8.
        path:  Destination file path. Parent directories are created
               automatically if they do not exist.

    Raises:
        ValueError: If OpenCV fails to encode or write the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise ValueError(
            f"OpenCV failed to write image to {path}. "
            f"Check that the extension is supported and the path is writable."
        )

    log.debug("write_image: saved %s  shape=%s", path.name, image.shape)


# ══════════════════════════════════════════════════════════════════════════
# Timing decorator
# ══════════════════════════════════════════════════════════════════════════

def timer(func: F) -> F:
    """
    Decorator that logs the wall-clock execution time of any function.

    Usage — add @timer above any function you want to profile:

        from infrastructure.utils import timer

        @timer
        def run_face_detection(frame):
            ...

    This will automatically log:
        [DEBUG] core_vision.face_detector — run_face_detection took 0.0312s

    The timing is logged at DEBUG level so it appears during development
    but can be silenced in production by raising the log level to INFO.

    Args:
        func: Any callable.

    Returns:
        The wrapped callable with identical signature and return type.
    """
    @functools.wraps(func)  # preserves __name__, __doc__, type hints
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Use the function's own module logger so the log line shows the
        # correct source module, not "infrastructure.utils"
        func_log = get_logger(func.__module__)
        func_log.debug("%s took %.4fs", func.__qualname__, elapsed)

        return result

    return wrapper  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════
# Array / tensor helpers
# ══════════════════════════════════════════════════════════════════════════

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 1-D vector so its magnitude equals 1.0.

    This is applied to face embeddings before computing cosine similarity.
    Two L2-normalised vectors have cosine_similarity = dot_product, which
    is faster to compute than the full cosine formula.

    Args:
        vector: np.ndarray of any shape, float dtype.

    Returns:
        L2-normalised copy of the input. If the norm is zero (zero vector),
        returns the original vector unchanged to avoid division by zero.
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        log.warning("l2_normalize received a zero vector — returning unchanged.")
        return vector
    return vector / norm


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image (OpenCV default) to RGB (PyTorch / PIL default).

    Use this at the boundary between OpenCV code and deep learning code.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image back to BGR.

    Use this when passing a model output back to OpenCV for display or saving.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)