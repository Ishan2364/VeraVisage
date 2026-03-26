"""
liveness/base_liveness_check.py
────────────────────────────────────────────────────────────────────────────
Abstract base class defining the interface every liveness check must follow.

WHY THIS EXISTS
───────────────
Every liveness check — texture, blink, optical flow, FFT — does one thing:
takes frames and returns a confidence score between 0 and 1.

By enforcing this contract through an ABC, the LivenessAggregator can hold
a list of ANY check objects and call .check() on all of them uniformly.
Adding a new liveness signal in the future means:
  1. Create a new class that inherits BaseLivenessCheck
  2. Implement check() and name
  3. Add it to the aggregator's list

Nothing else changes. This is the Open/Closed Principle in practice.
────────────────────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseLivenessCheck(ABC):
    """All liveness detectors must inherit from this class."""

    @abstractmethod
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Analyse the provided frames and return a liveness confidence score.

        Args:
            frames: List of BGR NumPy arrays (frames from the camera).
                    Some checks use only the first frame (spatial checks).
                    Others use all frames (temporal checks).
                    Shape of each frame: (H, W, 3), dtype uint8.

        Returns:
            Float in [0.0, 1.0]:
              1.0 = check is certain the input is a live person
              0.0 = check is certain the input is a spoof
              0.5 = check is uncertain / neutral

        Note:
            Implementations must NEVER raise exceptions for normal failure
            cases (no face detected, poor lighting). They should return 0.0
            or log a warning and return a neutral score. Exceptions are only
            appropriate for configuration errors (no model loaded, etc.).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this check.

        Used as the key in the aggregator's score breakdown dict.
        Example: "texture_lbp", "blink_ear", "optical_flow", "fft_artifact"
        """