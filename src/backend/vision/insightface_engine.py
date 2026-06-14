"""
InsightFace engine — lazy singleton wrapper around FaceAnalysis (buffalo_l).

Loads the model ONCE (first call), auto-selecting CUDAExecutionProvider when an
ONNX Runtime GPU build is available, otherwise CPUExecutionProvider. Exposes a
small, stable surface so the rest of the pipeline never touches InsightFace
internals directly:

    detect_faces(frame_bgr)               -> list[Face]
    get_face_embedding(face)              -> np.ndarray (aligned ArcFace embedding)
    extract_faces_with_embeddings(frame)  -> list[dict] {bbox, kps, det_score, embedding}

InsightFace internally detects, aligns to 112x112 using 5-point landmarks, and
produces the embedding, so we never feed raw bbox crops to the recognizer.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)

_face_app: Optional[Any] = None
_active_provider: Optional[str] = None


def _select_providers() -> List[str]:
    """Prefer CUDA when an onnxruntime-gpu build exposes it, else CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not query onnxruntime providers: {e}")
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_face_app():
    """Return the shared FaceAnalysis instance, initialising it on first use."""
    global _face_app, _active_provider
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis

    providers = _select_providers()
    use_gpu = providers[0] == "CUDAExecutionProvider"
    _active_provider = providers[0]

    logger.info(
        f"Loading InsightFace model pack '{config.MODEL_PACK}' "
        f"on {_active_provider} (det_size={config.DET_SIZE})..."
    )

    # Only load detection + recognition. The buffalo_l pack also ships dense
    # landmark (2d106/3d68) and gender-age models that FaceAnalysis would
    # otherwise run on EVERY detected face — pure overhead for our use case
    # (the detector already returns the 5 keypoints we need). Skipping them is a
    # large CPU speedup in crowded frames.
    app = FaceAnalysis(
        name=config.MODEL_PACK,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=config.DET_SIZE)

    _face_app = app
    logger.info(
        f"InsightFace ready. Provider: {_active_provider} "
        f"({'GPU' if use_gpu else 'CPU'})."
    )
    return _face_app


def active_provider() -> Optional[str]:
    """Provider in use once the model has been loaded (None before first use)."""
    return _active_provider


def detect_faces(frame_bgr: np.ndarray) -> List[Any]:
    """Detect faces in a BGR frame. Returns InsightFace Face objects."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []
    return list(get_face_app().get(frame_bgr))


def get_face_embedding(face: Any) -> Optional[np.ndarray]:
    """Return the aligned ArcFace embedding for a detected face."""
    emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def extract_faces_with_embeddings(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect faces and return normalised dicts for the pipeline.

    Each dict: {bbox: (x1,y1,x2,y2) int, kps: np.ndarray|None,
                det_score: float, embedding: np.ndarray|None}
    """
    faces = detect_faces(frame_bgr)
    results: List[Dict[str, Any]] = []
    for f in faces:
        bbox = getattr(f, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        results.append(
            {
                "bbox": (x1, y1, x2, y2),
                "kps": getattr(f, "kps", None),
                "det_score": float(getattr(f, "det_score", 0.0)),
                "embedding": get_face_embedding(f),
            }
        )
    return results
