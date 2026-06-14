"""
Detector abstraction (Phase 7 seam).

Selects the face detector backend via config.DETECTOR_BACKEND. Today only
InsightFace (SCRFD inside buffalo_l) is wired; SCRFD/RetinaFace standalone can
be added here later without changing the pipeline.
"""

from typing import Any, Dict, List

import numpy as np

from . import config


def extract_faces(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Return detected faces as dicts: {bbox, kps, det_score, embedding}.

    For the InsightFace backend the aligned ArcFace embedding is produced in the
    same pass, so it is attached here.
    """
    if config.DETECTOR_BACKEND == "insightface":
        from . import insightface_engine
        return insightface_engine.extract_faces_with_embeddings(frame_bgr)

    raise NotImplementedError(
        f"DETECTOR_BACKEND '{config.DETECTOR_BACKEND}' is not implemented yet"
    )
