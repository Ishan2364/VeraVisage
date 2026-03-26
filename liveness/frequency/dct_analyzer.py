"""
dct_analyzer.py
───────────────
DCT-domain analysis to catch JPEG/compression artifacts from deepfakes.
"""
import numpy as np
from liveness.base_liveness_check import BaseLivenessCheck


class DCTAnalyzer(BaseLivenessCheck):
    """Detects quantisation artefacts in the cosine-transform domain."""

    @property
    def name(self) -> str:
        return "dct_artifact"

    def check(self, frames: list[np.ndarray]) -> float:
        raise NotImplementedError
