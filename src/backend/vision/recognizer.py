"""
Recognizer abstraction (Phase 7 seam).

Selects the embedding backend via config.RECOGNIZER_BACKEND. The InsightFace
backend reuses the embedding already produced during detection. A future AdaFace
backend would re-embed the aligned face crop here; that seam is intentionally
left as NotImplementedError so it can be dropped in without touching callers.

(The "legacy" Haar + ResNet18 backend is a fully separate pipeline routed at the
SearchService level, not through this modern detect->embed path.)
"""

from typing import Any, Dict, Optional

import numpy as np

from . import config, matcher


def get_embedding(face: Dict[str, Any], frame_bgr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Return the face embedding for the configured recognizer backend."""
    if config.RECOGNIZER_BACKEND == "insightface":
        return face.get("embedding")

    if config.RECOGNIZER_BACKEND == "adaface":  # pragma: no cover - future
        raise NotImplementedError(
            "AdaFace recognizer backend is not implemented yet. "
            "Align the face crop to 112x112 and run the AdaFace model here."
        )

    raise NotImplementedError(
        f"RECOGNIZER_BACKEND '{config.RECOGNIZER_BACKEND}' is not implemented in the modern pipeline"
    )


def similarity(emb_a, emb_b) -> float:
    """Cosine similarity on normalised embeddings (delegates to matcher)."""
    return matcher.cosine(emb_a, emb_b)
