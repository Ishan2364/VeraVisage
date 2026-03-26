"""
core_vision/face_aligner.py
────────────────────────────────────────────────────────────────────────────
Aligns a detected face to a canonical 112×112 crop using the 5 landmarks
returned by face_detector.py.

WHY ALIGNMENT MATTERS
──────────────────────
ArcFace and FaceNet were trained on aligned faces. If you feed them an
unaligned crop (rotated head, off-centre eyes), the embedding vector will
be significantly worse — cosine similarities between the same person
drop from ~0.85 to ~0.55, making reliable verification impossible.

WHAT ALIGNMENT DOES
────────────────────
Given 5 landmark points from the detector:
    left_eye, right_eye, nose, left_mouth, right_mouth

We compute an affine transform (rotation + scale + translation — no shear,
no perspective) that maps those 5 points to 5 fixed reference positions
on a 112×112 canvas. The reference positions were chosen so that:
    - Eyes land at y≈37, horizontally centred
    - Nose lands at the centre
    - Mouth lands at y≈75

The result is a tightly cropped, upright face regardless of how the person
is positioned in the original frame.

REFERENCE LANDMARKS
────────────────────
These are the ArcFace/InsightFace standard 112×112 reference positions.
Using these (rather than inventing your own) ensures compatibility with
pretrained ArcFace weights.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer

log = get_logger(__name__)

# ── ArcFace standard 112×112 reference landmark positions ─────────────────
# Order matches insightface output: left_eye, right_eye, nose, left_mouth, right_mouth
# These are the target pixel coordinates AFTER alignment.
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

OUTPUT_SIZE = (112, 112)  # standard ArcFace input size


class FaceAligner:
    """
    Aligns detected face crops to the ArcFace 112×112 canonical pose.

    Usage:
        aligner = FaceAligner()
        aligned = aligner.align(frame, landmarks)
        # aligned.shape == (112, 112, 3), BGR, ready for embedding_generator
    """

    def __init__(self, output_size: tuple[int, int] = OUTPUT_SIZE):
        """
        Args:
            output_size: (width, height) of the aligned output crop.
                         112×112 is the ArcFace standard. Only change
                         this if you switch to a different backbone that
                         expects a different input size.
        """
        self.output_size = output_size
        log.info("FaceAligner initialised — output_size=%s", output_size)

    @timer
    def align(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Warp the face in `frame` so its landmarks match the reference positions.

        Args:
            frame:     Full BGR frame from the camera, shape (H, W, 3).
            landmarks: 5 landmark points from face_detector.detect(),
                       shape (5, 2), dtype float32.
                       Order: left_eye, right_eye, nose, left_mouth, right_mouth.

        Returns:
            Aligned face crop, shape (112, 112, 3), dtype uint8, BGR.
            Ready to pass directly into embedding_generator.py.

        Raises:
            ValueError: If landmarks shape is wrong or transform estimation fails.
        """
        if landmarks.shape != (5, 2):
            raise ValueError(
                f"landmarks must have shape (5, 2), got {landmarks.shape}. "
                f"Ensure you're passing the 'landmarks' key from face_detector output."
            )

        # Scale reference landmarks if output_size differs from 112×112
        ref = REFERENCE_LANDMARKS.copy()
        if self.output_size != (112, 112):
            scale_x = self.output_size[0] / 112.0
            scale_y = self.output_size[1] / 112.0
            ref[:, 0] *= scale_x
            ref[:, 1] *= scale_y

        # estimateAffinePartial2D finds the best-fit rotation + scale +
        # translation that maps `landmarks` → `ref`.
        # LMEDS is more robust to outlier landmarks than the default RANSAC.
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            landmarks,
            ref,
            method=cv2.LMEDS,
        )

        if transform_matrix is None:
            raise ValueError(
                "Affine transform estimation failed. "
                "The landmarks may be degenerate (e.g. all points collinear). "
                "Try a clearer, more frontal face image."
            )

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        log.debug(
            "Affine transform estimated — inliers=%d/5  matrix=\n%s",
            inlier_count, transform_matrix,
        )

        # Apply the transform to the full frame.
        # warpAffine interpolates with INTER_LINEAR (bilinear) by default —
        # good balance of quality and speed.
        aligned = cv2.warpAffine(
            frame,
            transform_matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
            # REFLECT_101 fills border pixels by mirroring the edge content,
            # which looks more natural than black padding for face crops.
        )

        log.debug("Aligned face — output shape=%s", aligned.shape)
        return aligned

    def align_from_detection(
        self,
        frame: np.ndarray,
        detection: dict,
    ) -> np.ndarray:
        """
        Convenience wrapper — takes the full detection dict from face_detector
        and calls align() with the landmarks inside it.

        Args:
            frame:     Full BGR frame.
            detection: Dict with keys "bbox", "landmarks", "confidence"
                       as returned by FaceDetector.detect_single().

        Returns:
            Aligned face crop, shape (112, 112, 3), BGR.
        """
        return self.align(frame, detection["landmarks"])