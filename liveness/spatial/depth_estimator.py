"""
depth_estimator.py
──────────────────
Monocular depth-cue analysis to detect flat 2-D spoof surfaces.
"""
import numpy as np
from liveness.base_liveness_check import BaseLivenessCheck


class DepthEstimator(BaseLivenessCheck):
    """Estimates depth variation across the face region."""

    @property
    def name(self) -> str:
        return "depth_estimation"

    def check(self, frames: list[np.ndarray]) -> float:
        raise NotImplementedError
