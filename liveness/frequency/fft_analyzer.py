"""
fft_analyzer.py
───────────────
2-D FFT to detect GAN frequency artifacts and screen moiré patterns.
"""
import numpy as np
from liveness.base_liveness_check import BaseLivenessCheck


class FFTAnalyzer(BaseLivenessCheck):
    """Identifies high-frequency spectral anomalies introduced by GAN synthesis."""

    @property
    def name(self) -> str:
        return "fft_artifact"

    def check(self, frames: list[np.ndarray]) -> float:
        raise NotImplementedError
