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


def similarity_to_percent(sim: float) -> float:
    """Map a cosine similarity to a calibrated 0-100 'match accuracy' %.

    Cosine similarity is NOT a linear probability: genuine matches score
    ~0.55-0.63 while imposters cap at ~0.49, so a naive ``sim * 100`` makes a
    strong match read as a misleadingly low number. Instead this is a
    piecewise-linear mapping anchored on the decision thresholds so a
    strong-match cosine reads high and a possible-match cosine reads moderate.
    Anchors are read from config so DRISHTI_* threshold overrides stay
    consistent.
    """
    anchors = [
        (0.0, 0.0),
        (config.POSSIBLE_MATCH_THRESHOLD, 70.0),
        (config.MATCH_THRESHOLD, 85.0),
        (1.0, 100.0),
    ]
    # Keep anchors strictly increasing in x (guards against odd threshold overrides).
    clean = [anchors[0]]
    for x, y in anchors[1:]:
        if x > clean[-1][0]:
            clean.append((x, y))
    s = max(0.0, min(1.0, float(sim)))
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        if s <= x1:
            t = (s - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 1)
    return 100.0


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
