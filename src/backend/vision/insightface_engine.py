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
import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _register_cuda_dll_dirs() -> None:
    """Put the pip-installed CUDA/cuDNN ``nvidia/*/bin`` dirs on the Windows DLL
    search path.

    ``preload_dlls()`` loads the top-level DLLs (``cudnn64_9.dll`` etc.), but
    cuDNN then loads its OWN sublibraries (``cudnn_graph64_9.dll``, the engine
    DLLs) at runtime via the default search order — which does NOT include the
    wheels' ``nvidia/cudnn/bin`` dir. Without this registration that lazy load
    fails with ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` and ORT falls back to
    CPU. Best-effort: a no-op when the wheels (or ``add_dll_directory``) are
    absent, so CPU-only installs are unaffected.
    """
    if not hasattr(os, "add_dll_directory"):  # non-Windows
        return
    try:
        import nvidia  # provided by the nvidia-*-cu12 wheels
    except Exception:
        return
    base = os.path.dirname(nvidia.__file__)
    try:
        subdirs = os.listdir(base)
    except OSError:
        return
    for sub in subdirs:
        bin_dir = os.path.join(base, sub, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass


def _preload_cuda_dlls() -> None:
    """Preload the CUDA/cuDNN DLLs shipped by the ``nvidia-*-cu12`` pip wheels so
    onnxruntime-gpu's CUDAExecutionProvider can initialise.

    onnxruntime-gpu's ``onnxruntime_providers_cuda.dll`` links against
    ``cublasLt64_12.dll`` / ``cudnn*64_9.dll``; on Windows it does not search the
    pip wheels' ``nvidia/*/bin`` dirs for these transitive dependencies, so the
    provider fails to load and ORT silently falls back to CPU. ``preload_dlls()``
    (onnxruntime-gpu >= 1.21) loads them explicitly from the installed wheels.

    Best-effort and self-contained: on a CPU-only install ``preload_dlls`` is
    absent or finds nothing, so this is a no-op and search runs on CPU as before.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"onnxruntime.preload_dlls() skipped: {e}")


# Must run before any InferenceSession is created (i.e. before InsightFace loads
# its ONNX models), otherwise the CUDA provider cannot find its dependencies.
# Order matters: register the search dirs first so cuDNN's lazily-loaded
# sublibraries resolve, then preload the top-level CUDA/cuDNN DLLs.
_register_cuda_dll_dirs()
_preload_cuda_dlls()

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
