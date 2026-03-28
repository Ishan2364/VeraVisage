"""
liveness/active/reflection_analyzer.py
────────────────────────────────────────────────────────────────────────────
Analyses facial skin colour response to the flash challenge using
3D Lambertian reflection checks and HSV Material Physics.

THE HSV GLARE TRAP (Anti-Glass Exploit)
────────────────────────────────────────
Instead of just checking if the skin "changed colour", we translate the 
pixels into the HSV (Hue, Saturation, Value) colour space to test the 
material properties of the face:

1. Value (Brightness) Blowout: 
   Human skin diffuses light softly. Phone glass is a mirror (specular).
   If the brightness (Value) spikes massively, it's a glass mirror.
   
2. Saturation Washout:
   Skin absorbs light and retains its pigment. Glass glare turns pure white,
   destroying the underlying picture's colour. If Saturation plummets, it's glass.

3. The 3D Shadow Variance:
   The variance in brightness between the forehead and the cheeks must fall
   in a "Goldilocks" zone. Too uniform = flat phone. Too varied = sharp glare band.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import numpy as np

from infrastructure.logger import get_logger
from infrastructure.utils import timer

log = get_logger(__name__)

# ── Physical Optical Thresholds ───────────────────────────────────────────
MIN_ALIGNMENT_SCORE  = 0.40   # Colour hue must loosely match the flash

# HSV Brightness (Value) Gates (Range: 0-255)
MIN_V_SHIFT          = 5.0    # Must respond to the light (Not printed paper)
MAX_V_SHIFT          = 80.0   # If brightness jumps 80+ points, it's a glass mirror

# HSV Saturation Gates (Range: -255 to 255)
MAX_S_DROP           = -60.0  # If saturation drops by 60+ points, glare washed it out

# 3D Geometry Gates (Ratio of min_brightness_shift / max_brightness_shift)
MAX_PLANAR_RATIO     = 0.85   # > 0.85 means perfectly flat surface (Phone facing straight)
MIN_PLANAR_RATIO     = 0.15   # < 0.15 means one spot is blinded while others are dark (Harsh glare band)

MIN_COLOURS_PASSING  = 2      # out of SEQUENCE_LENGTH colours must pass

# Skin ROI fractions of face bounding box
FOREHEAD = dict(top=0.05, bottom=0.28, left=0.20, right=0.80)
L_CHEEK  = dict(top=0.40, bottom=0.65, left=0.10, right=0.38)
R_CHEEK  = dict(top=0.40, bottom=0.65, left=0.62, right=0.90)


class ReflectionAnalyzer:
    def __init__(
        self,
        min_alignment: float = MIN_ALIGNMENT_SCORE,
        min_colours_passing: int = MIN_COLOURS_PASSING,
        face_detector=None,
    ):
        self.min_alignment       = min_alignment
        self.min_colours_passing = min_colours_passing
        self.face_detector       = face_detector

        log.info("ReflectionAnalyzer initialized with HSV Material Physics.")

    def _extract_skin_parts(
        self,
        frame: np.ndarray,
        bbox: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extracts skin ROIs separately to check for 3D depth variance."""
        h, w = frame.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fw, fh = x2 - x1, y2 - y1
        else:
            x1, x2 = int(w * 0.20), int(w * 0.80)
            y1, y2 = int(h * 0.10), int(h * 0.90)
            fw, fh = x2 - x1, y2 - y1

        def get_roi(region: dict) -> np.ndarray:
            ry1 = max(0, y1 + int(fh * region["top"]))
            ry2 = min(h, y1 + int(fh * region["bottom"]))
            rx1 = max(0, x1 + int(fw * region["left"]))
            rx2 = min(w, x1 + int(fw * region["right"]))
            if ry2 <= ry1 or rx2 <= rx1:
                return np.array([]).reshape(0, 3)
            return frame[ry1:ry2, rx1:rx2].reshape(-1, 3).astype(np.float32)

        return {
            "forehead": get_roi(FOREHEAD),
            "l_cheek":  get_roi(L_CHEEK),
            "r_cheek":  get_roi(R_CHEEK),
        }

    def _mean_skin_parts(
        self,
        frames: list[np.ndarray],
        bbox: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Computes mean BGR for each face part across frames."""
        parts_data = {"forehead": [], "l_cheek": [], "r_cheek": []}
        
        for frame in frames:
            parts = self._extract_skin_parts(frame, bbox)
            for key, pixels in parts.items():
                if len(pixels) > 0:
                    parts_data[key].append(pixels)

        result = {}
        for key, pixels_list in parts_data.items():
            if pixels_list:
                result[key] = np.vstack(pixels_list).mean(axis=0)
            else:
                result[key] = np.array([0.0, 0.0, 0.0])
        return result

    def _bgr_to_hsv(self, bgr_array: np.ndarray) -> np.ndarray:
        """Converts a 1D mean BGR array [B, G, R] to HSV [H, S, V]."""
        # Constrain to valid 8-bit image limits for OpenCV conversion
        bgr_uint8 = np.clip(bgr_array, 0, 255).astype(np.uint8)
        pixel = np.array([[bgr_uint8]])
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        return hsv_pixel[0][0].astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def analyze_single_flash(
        self,
        baseline_parts: dict[str, np.ndarray],
        flash_frames: list[np.ndarray],
        flash_bgr: tuple[int, int, int],
        bbox: list[int] | None = None,
    ) -> dict:
        
        flash_parts = self._mean_skin_parts(flash_frames, bbox)
        
        bgr_shifts = []
        v_shifts = []
        s_shifts = []
        
        for key in ["forehead", "l_cheek", "r_cheek"]:
            # BGR for colour direction
            bgr_shift = flash_parts[key] - baseline_parts[key]
            bgr_shifts.append(bgr_shift)

            # Convert means to HSV for material physics checks
            base_hsv = self._bgr_to_hsv(baseline_parts[key])
            flash_hsv = self._bgr_to_hsv(flash_parts[key])

            # Value (Brightness) is index 2. Saturation is index 1.
            v_shift = flash_hsv[2] - base_hsv[2]
            s_shift = flash_hsv[1] - base_hsv[1]
            
            v_shifts.append(v_shift)
            s_shifts.append(s_shift)

        # Colour Alignment Check (Do the pixels move toward the flash colour?)
        overall_bgr_shift = np.mean(bgr_shifts, axis=0)
        expected_bgr = np.array(flash_bgr, dtype=np.float32)
        alignment = self._cosine_similarity(overall_bgr_shift, expected_bgr)

        # Extract 3D Variance Metrics
        valid_v_shifts = [v for v in v_shifts if v > 0]
        if len(valid_v_shifts) >= 2:
            min_v = min(valid_v_shifts)
            max_v = max(valid_v_shifts)
            planar_ratio = min_v / max_v
        else:
            min_v = 0.0
            max_v = max(v_shifts) if v_shifts else 0.0
            planar_ratio = 1.0

        overall_v_shift = float(np.mean(v_shifts))
        max_s_drop = float(min(s_shifts)) # Find the region that lost the most colour

        # ─── THE HSV GLARE TRAP GATES ─────────────────────────────────────
        
        # 1. Did it respond to light? (Paper veto)
        is_responsive = overall_v_shift >= MIN_V_SHIFT
        
        # 2. Is it a mirror blowout? (Glass veto)
        is_matte_v = max_v <= MAX_V_SHIFT
        
        # 3. Did the glare wash out the colour? (Glass washout veto)
        is_matte_s = max_s_drop >= MAX_S_DROP 
        
        # 4. Is it perfectly flat? (Phone held straight veto)
        is_3d = planar_ratio <= MAX_PLANAR_RATIO
        
        # 5. Is there a harsh glare band? (Phone held tilted veto)
        no_harsh_glare = planar_ratio >= MIN_PLANAR_RATIO

        passed = (
            alignment >= self.min_alignment and
            is_responsive and
            is_matte_v and
            is_matte_s and
            is_3d and
            no_harsh_glare
        )

        log.debug(
            "Flash %s: align=%.2f, V_shift=%.1f (max=%.1f), S_drop=%.1f, ratio=%.2f, passed=%s",
            flash_bgr, alignment, overall_v_shift, max_v, max_s_drop, planar_ratio, passed,
        )

        return {
            "passed":       passed,
            "alignment":    alignment,
            "magnitude":    overall_v_shift, # using V_shift as magnitude now
            "planar_ratio": planar_ratio,
        }

    @timer
    def analyze(
        self,
        challenge_result: dict,
        bbox: list[int] | None = None,
    ) -> tuple[bool, float, dict]:
        
        if not challenge_result.get("success", False):
            return False, 0.0, {"error": "insufficient frames"}

        baseline_frames = challenge_result["baseline_frames"]
        flash_data      = challenge_result["flash_data"]

        baseline_parts = self._mean_skin_parts(baseline_frames, bbox)

        results = []
        breakdown = {}
        colours_passing = 0

        for fd in flash_data:
            result = self.analyze_single_flash(
                baseline_parts=baseline_parts,
                flash_frames=fd["frames"],
                flash_bgr=fd["color_bgr"],
                bbox=bbox,
            )
            results.append(result)
            breakdown[fd["color_name"]] = {
                "passed":       result["passed"],
                "alignment":    round(result["alignment"], 3),
                "v_shift":      round(result["magnitude"], 2),
                "planar_ratio": round(result["planar_ratio"], 2),
            }
            if result["passed"]:
                colours_passing += 1

        n_colours = max(1, len(flash_data))
        pass_rate = colours_passing / n_colours
        mean_alignment = float(np.mean([r["alignment"] for r in results]))

        # Final Score Math
        score = float(np.clip(
            0.70 * pass_rate + 0.30 * max(0.0, mean_alignment),
            0.0, 1.0,
        ))

        is_live = colours_passing >= self.min_colours_passing

        breakdown["summary"] = {
            "colours_passing":  colours_passing,
            "pass_rate":        round(pass_rate, 3),
            "score":            round(score, 3),
            "is_live":          is_live,
        }

        return is_live, score, breakdown