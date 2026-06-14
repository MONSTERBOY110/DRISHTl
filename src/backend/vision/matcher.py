"""
Similarity + open-set decision logic.

Cosine similarity on L2-normalised embeddings. Cosine similarity is NOT a
percentage confidence and must never be presented as one. The decision is made
per TRACK (a person followed across frames), never from a single frame, and the
system is allowed to return "no_reliable_match".
"""

from typing import Dict, List, Optional

import numpy as np

from . import config


def normalize(embedding) -> Optional[np.ndarray]:
    """Return the L2-normalised float32 embedding, or None if invalid/zero."""
    if embedding is None:
        return None
    e = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(e))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return e / norm


def cosine(a, b) -> float:
    """Cosine similarity in [-1, 1]; -1.0 if either embedding is invalid."""
    na, nb = normalize(a), normalize(b)
    if na is None or nb is None:
        return -1.0
    return float(np.dot(na, nb))


def decide_track_match(similarities: List[float]) -> Dict[str, object]:
    """
    Open-set decision for one track given its per-frame similarities.

    strong_match:
        >= MIN_MATCH_FRAMES frames clear MATCH_THRESHOLD, AND the average of the
        top strong scores clears MIN_AVG_SIMILARITY, AND best >= MATCH_THRESHOLD.
    possible_match:
        >= MIN_MATCH_FRAMES + 1 frames clear POSSIBLE_MATCH_THRESHOLD (but not
        strong).
    otherwise:
        no_reliable_match.
    """
    if not similarities:
        return {"decision": "no_reliable_match", "avg_similarity": 0.0,
                "best_similarity": 0.0, "matched_frames": 0}

    best = max(similarities)
    strong = [s for s in similarities if s >= config.MATCH_THRESHOLD]

    if len(strong) >= config.MIN_MATCH_FRAMES and best >= config.MATCH_THRESHOLD:
        top = sorted(strong, reverse=True)[:5]
        avg_top = sum(top) / len(top)
        if avg_top >= config.MIN_AVG_SIMILARITY:
            return {
                "decision": "strong_match",
                "avg_similarity": round(avg_top, 4),
                "best_similarity": round(best, 4),
                "matched_frames": len(strong),
            }

    possible = [s for s in similarities if s >= config.POSSIBLE_MATCH_THRESHOLD]
    if len(possible) >= config.MIN_MATCH_FRAMES + 1:
        return {
            "decision": "possible_match",
            "avg_similarity": round(sum(possible) / len(possible), 4),
            "best_similarity": round(best, 4),
            "matched_frames": len(possible),
        }

    return {
        "decision": "no_reliable_match",
        "avg_similarity": round(sum(similarities) / len(similarities), 4),
        "best_similarity": round(best, 4),
        "matched_frames": 0,
    }
