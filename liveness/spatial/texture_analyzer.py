"""
liveness/spatial/texture_analyzer.py
────────────────────────────────────────────────────────────────────────────
Texture-based liveness detection using Local Binary Patterns (LBP) + SVM.

HOW IT WORKS
────────────
1. Take the aligned 112×112 face crop from face_aligner.py
2. Convert to grayscale
3. Compute LBP (Local Binary Pattern) descriptor over the face
4. Flatten into a histogram feature vector
5. Feed into a pre-trained SVM classifier
6. SVM outputs: 1 = live face, 0 = spoof (printed photo)

WHY LBP WORKS FOR SPOOFING
────────────────────────────
A real face has complex 3D skin micro-texture — pores, fine hair, subtle
depth variation. A printed photo flattens all of this. The ink pattern
on paper produces a fundamentally different LBP histogram than real skin.
LBP captures exactly this — it encodes the relationship between each pixel
and its neighbours as a binary pattern, making it sensitive to the kind of
texture differences that distinguish skin from paper.

WHAT IT CANNOT CATCH
─────────────────────
High-quality screen replays and 3D masks can fool texture analysis because
their surface texture is more complex than a flat print. That is why we
combine this with blink detection, optical flow, and FFT analysis.

USAGE
─────
# Training (run once via scripts/train_texture_svm.py):
    analyzer = TextureAnalyzer()
    analyzer.train(real_images, spoof_images)
    analyzer.save_model("data/models/texture_svm.pkl")

# Inference (called by liveness_aggregator.py):
    analyzer = TextureAnalyzer()
    analyzer.load_model("data/models/texture_svm.pkl")
    score = analyzer.check([aligned_frame])  # returns float in [0, 1]
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from infrastructure.exceptions import DeepfakeAuthError
from infrastructure.logger import get_logger
from infrastructure.utils import timer
from liveness.base_liveness_check import BaseLivenessCheck

log = get_logger(__name__)

# ── LBP hyperparameters ───────────────────────────────────────────────────
# These are the standard values used in anti-spoofing literature.
# P = number of circularly symmetric neighbour set points
# R = radius of circle (in pixels)
# METHOD = "uniform" keeps only the 58 most common patterns + 1 non-uniform
#          bin, giving a compact 59-bin histogram per cell
LBP_P = 8
LBP_R = 1.0
LBP_METHOD = "uniform"

# We divide the face into a grid of cells and compute LBP per cell,
# then concatenate. This preserves spatial layout (eyes vs nose vs mouth
# have different textures) and significantly improves accuracy over a
# single global LBP histogram.
GRID_X = 8   # horizontal cells
GRID_Y = 8   # vertical cells


class TextureAnalyzer(BaseLivenessCheck):
    """
    LBP + SVM texture-based spoof detector.

    Implements BaseLivenessCheck so it plugs directly into
    LivenessAggregator without any glue code.
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Args:
            model_path: Optional path to a pre-trained SVM .pkl file.
                        If provided, the model is loaded immediately.
                        If None, call load_model() or train() before check().
        """
        self._svm = None
        self.model_path = Path(model_path) if model_path else None

        if self.model_path and self.model_path.exists():
            self.load_model(self.model_path)
        elif self.model_path:
            log.warning(
                "TextureAnalyzer: model_path provided but file not found: %s. "
                "Run scripts/train_texture_svm.py first.",
                self.model_path,
            )

    @property
    def name(self) -> str:
        return "texture_lbp"

    # ── Feature extraction ────────────────────────────────────────────────

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a grid LBP feature vector from a face crop.

        Args:
            image: BGR face crop, ideally 112×112 from face_aligner.py.
                   Other sizes work but 112×112 is optimal.

        Returns:
            1-D float32 feature vector of length GRID_X * GRID_Y * n_bins
            where n_bins = LBP_P + 2 = 10 for uniform LBP with P=8.
        """
        # Convert to grayscale — LBP operates on intensity, not colour
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to a fixed size so feature vectors are always the same
        # length regardless of input resolution
        gray = cv2.resize(gray, (112, 112))

        cell_h = gray.shape[0] // GRID_Y
        cell_w = gray.shape[1] // GRID_X

        histograms = []

        for row in range(GRID_Y):
            for col in range(GRID_X):
                # Extract one cell
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                cell = gray[y1:y2, x1:x2]

                # Compute LBP for this cell
                lbp = local_binary_pattern(cell, P=LBP_P, R=LBP_R, method=LBP_METHOD)

                # Build normalised histogram
                n_bins = LBP_P + 2  # uniform LBP has P+2 bins
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=n_bins,
                    range=(0, n_bins),
                    density=True,  # normalise so lighting doesn't dominate
                )
                histograms.append(hist)

        # Concatenate all cell histograms into one feature vector
        feature_vector = np.concatenate(histograms).astype(np.float32)
        return feature_vector

    def extract_features_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of images.

        Returns:
            2-D array of shape (N, feature_dim).
        """
        return np.array([self.extract_features(img) for img in images])

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        real_images: list[np.ndarray],
        spoof_images: list[np.ndarray],
    ) -> dict:
        """
        Train the SVM on real vs spoof face crops.

        Args:
            real_images:  List of BGR face crops of real/live faces.  label=1
            spoof_images: List of BGR face crops of spoofed faces.    label=0

        Returns:
            Dict with training accuracy and number of samples.
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        log.info(
            "Training TextureAnalyzer SVM — real=%d, spoof=%d samples",
            len(real_images), len(spoof_images),
        )

        # Extract features
        X_real  = self.extract_features_batch(real_images)
        X_spoof = self.extract_features_batch(spoof_images)
        X = np.vstack([X_real, X_spoof])
        y = np.array([1] * len(real_images) + [0] * len(spoof_images))

        # Pipeline: StandardScaler normalises the feature range before SVM
        # RBF kernel SVM is the standard choice for LBP anti-spoofing
        self._svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                          probability=True,   # needed for predict_proba()
                          class_weight="balanced")),
        ])

        # 5-fold cross-validation to report honest accuracy
        cv_scores = cross_val_score(self._svm, X, y, cv=5, scoring="accuracy")
        log.info(
            "Cross-validation accuracy: %.3f ± %.3f",
            cv_scores.mean(), cv_scores.std(),
        )

        # Fit on full dataset
        self._svm.fit(X, y)

        train_acc = self._svm.score(X, y)
        log.info("Training accuracy (full set): %.3f", train_acc)

        return {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std":  float(cv_scores.std()),
            "train_accuracy":   float(train_acc),
            "n_real":           len(real_images),
            "n_spoof":          len(spoof_images),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save_model(self, path: str | Path) -> None:
        """Save the trained SVM pipeline to disk as a .pkl file."""
        if self._svm is None:
            raise DeepfakeAuthError("No model to save — train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._svm, f)
        log.info("TextureAnalyzer model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved SVM pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Texture SVM model not found at {path}. "
                f"Run scripts/train_texture_svm.py first."
            )
        with open(path, "rb") as f:
            self._svm = pickle.load(f)
        log.info("TextureAnalyzer model loaded from %s", path)

    # ── Inference (BaseLivenessCheck interface) ───────────────────────────

    @timer
    def check(self, frames: list[np.ndarray]) -> float:
        """
        Run texture liveness check on a list of frames.

        Uses the first frame only (texture is a spatial check, not temporal).
        Averages predictions over up to 5 frames for stability.

        Args:
            frames: List of BGR face crops from face_aligner.py.

        Returns:
            Liveness confidence in [0.0, 1.0].
            1.0 = definitely live skin texture.
            0.0 = definitely spoof texture (printed paper).

        Raises:
            DeepfakeAuthError: If no model is loaded.
        """
        if self._svm is None:
            raise DeepfakeAuthError(
                "TextureAnalyzer has no loaded model. "
                "Call load_model() or train() first."
            )

        if not frames:
            log.warning("TextureAnalyzer.check() received empty frames list")
            return 0.0

        # Use up to 5 frames and average for stability
        sample_frames = frames[:5]
        scores = []

        for frame in sample_frames:
            features = self.extract_features(frame).reshape(1, -1)
            # predict_proba returns [P(spoof), P(live)]
            proba = self._svm.predict_proba(features)[0]
            live_score = float(proba[1])  # index 1 = class "1" = live
            scores.append(live_score)

        final_score = float(np.mean(scores))
        log.debug("TextureAnalyzer score: %.3f (avg over %d frames)", final_score, len(scores))
        return final_score