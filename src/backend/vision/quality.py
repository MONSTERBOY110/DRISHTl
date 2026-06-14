"""
Face quality gate.

Rejects weak/ambiguous face crops BEFORE they ever reach the recognizer, so
background regions, tiny faces, blurry faces and bad exposure can never produce
an accidental high-similarity match. Every rejection maps to a named counter
that is surfaced in the API response for debugging.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from . import config

# Rejection reason -> counter key returned in the API "rejected" object.
REJECT_KEYS = (
    "low_detection_score",
    "small_faces",
    "blurry_faces",
    "bad_brightness",
    "missing_landmarks",
)


def new_reject_counter() -> Dict[str, int]:
    """Fresh zeroed counter dict for an entire search run."""
    return {k: 0 for k in REJECT_KEYS}


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    """Mean grayscale intensity (0-255)."""
    return float(np.mean(gray))


def _crop(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def is_valid_face(
    face: Dict[str, Any],
    frame_bgr: np.ndarray,
    *,
    min_det: Optional[float] = None,
    min_size: Optional[int] = None,
    min_blur: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a detected face dict (as produced by insightface_engine).

    Thresholds default to the strict VIDEO-frame gate (config.MIN_*); callers may
    override per-check (the target photo uses the lenient TARGET_MIN_* values).

    Returns (ok, reject_reason). reject_reason is one of REJECT_KEYS when ok is
    False, else None.
    """
    min_det = config.MIN_DETECTION_SCORE if min_det is None else min_det
    min_size = config.MIN_FACE_SIZE if min_size is None else min_size
    min_blur = config.MIN_BLUR_SCORE if min_blur is None else min_blur

    # 1. Detection confidence
    if face.get("det_score", 0.0) < min_det:
        return False, "low_detection_score"

    # 2. Landmarks present (InsightFace gives 5 keypoints)
    kps = face.get("kps")
    if kps is None or len(kps) < 5:
        return False, "missing_landmarks"

    # 3. Face size
    x1, y1, x2, y2 = face["bbox"]
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False, "small_faces"

    crop = _crop(frame_bgr, face["bbox"])
    if crop is None:
        return False, "small_faces"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 4. Blur
    if compute_blur_score(gray) < min_blur:
        return False, "blurry_faces"

    # 5. Brightness (too dark / overexposed)
    brightness = compute_brightness(gray)
    if brightness < config.MIN_BRIGHTNESS or brightness > config.MAX_BRIGHTNESS:
        return False, "bad_brightness"

    return True, None
