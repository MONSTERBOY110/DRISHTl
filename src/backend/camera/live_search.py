"""
Live-buffer search.

Compares an uploaded target face against the embeddings indexed in a camera's
rolling buffer, restricted to the LAST `duration_seconds` only — so a person who
left the frame more than `duration_seconds` ago is NOT returned (no stale
matches, no reuse of old snapshots). Returns the same result shape as the static
pipeline so the frontend reuses its result cards.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config.settings import Settings
from src.backend.vision import config, detector, matcher, quality
from src.backend.camera.camera_manager import camera_manager

logger = logging.getLogger(__name__)

WARMING_MESSAGE = "Live camera buffer is still warming up. Please wait a few seconds."


def _no_recent_message(duration: int) -> str:
    return f"No reliable match found in the last {duration}s of the live camera feed."


def _target_embedding(target_path: str):
    img = cv2.imread(target_path)
    if img is None:
        return None, "Could not read uploaded image"
    faces = detector.extract_faces(img)
    if not faces:
        return None, "No face detected in the uploaded photo."
    best, bw = None, 0
    for f in faces:
        ok, _ = quality.is_valid_face(
            f, img, min_det=config.TARGET_MIN_DETECTION_SCORE,
            min_size=config.TARGET_MIN_FACE_SIZE, min_blur=config.TARGET_MIN_BLUR_SCORE)
        if ok and (f["bbox"][2] - f["bbox"][0]) > bw:
            bw = f["bbox"][2] - f["bbox"][0]; best = f
    if best is None:
        return None, "The face in the uploaded photo is too low quality to search reliably."
    emb = matcher.normalize(best.get("embedding"))
    if emb is None:
        return None, "Could not extract a usable face embedding from the photo."
    return emb, None


def _save(jpeg_bytes: bytes, filename: str) -> bool:
    try:
        Settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(Settings.RESULTS_DIR / filename, "wb") as fh:
            fh.write(jpeg_bytes)
        return True
    except Exception as e:  # best-effort
        logger.warning(f"live snapshot save failed: {e}")
        return False


def search_live_buffer(camera_id: str, target_path: str, search_id: str,
                       duration_seconds: int = None) -> Dict[str, Any]:
    duration_seconds = int(duration_seconds or config.LIVE_SEARCH_DURATION_SECONDS)

    worker = camera_manager.get(camera_id)
    if worker is None:
        return {"search_id": search_id, "status": "error", "decision": "no_reliable_match",
                "matches": [], "message": f"Camera '{camera_id}' is not connected."}

    target_emb, err = _target_embedding(target_path)
    if target_emb is None:
        return {"search_id": search_id, "status": "completed", "decision": "no_reliable_match",
                "matches": [], "message": err}

    search_now = time.time()
    within, total_entries, pruned = worker.buffer.recent(duration_seconds, search_now)
    latest_ts = worker.buffer.latest_timestamp()
    latest_age = round(search_now - latest_ts, 2) if latest_ts else None

    def live_debug(reason: str, considered: int) -> Dict[str, Any]:
        return {
            "camera_id": camera_id,
            "search_duration_seconds": duration_seconds,
            "buffer_total_entries": total_entries,
            "entries_considered": considered,
            "old_entries_pruned": pruned,
            "latest_buffer_age_seconds": latest_age,
            "latest_processed_frame_time":
                datetime.fromtimestamp(latest_ts).strftime("%H:%M:%S") if latest_ts else None,
            "decision_reason": reason,
        }

    if total_entries == 0:
        return {"search_id": search_id, "status": "completed", "decision": "no_reliable_match",
                "matches": [], "message": WARMING_MESSAGE,
                "live_debug": live_debug("buffer_empty", 0)}

    if not within:
        return {"search_id": search_id, "status": "completed", "decision": "no_reliable_match",
                "matches": [], "message": _no_recent_message(duration_seconds),
                "live_debug": live_debug("no_recent_frames", 0)}

    # Compare target only to faces from the recent window.
    sims: List[float] = []
    matched_ts: List[float] = []
    best = {"sim": -1.0, "entry": None, "bbox": None}
    faces_compared = 0
    for entry in within:
        frame_best = -1.0
        for face in entry["faces"]:
            faces_compared += 1
            s = matcher.cosine(target_emb, face["embedding"])
            if s > frame_best:
                frame_best = s
            if s > best["sim"]:
                best = {"sim": s, "entry": entry, "bbox": face["bbox"]}
        if frame_best >= 0:
            sims.append(frame_best)
            if frame_best >= config.POSSIBLE_MATCH_THRESHOLD:
                matched_ts.append(entry["timestamp"])

    verdict = matcher.decide_track_match(sims) if sims else {
        "decision": "no_reliable_match", "best_similarity": 0.0, "avg_similarity": 0.0, "matched_frames": 0}

    if verdict["decision"] == "no_reliable_match":
        return {"search_id": search_id, "status": "completed", "decision": "no_reliable_match",
                "matches": [], "message": _no_recent_message(duration_seconds),
                "live_debug": live_debug("no_recent_match", len(within))}

    # Staleness guard: the best frame must be inside the requested window.
    best_ts = best["entry"]["timestamp"]
    age_seconds = round(search_now - best_ts, 2)
    if age_seconds > duration_seconds:
        return {"search_id": search_id, "status": "completed", "decision": "no_reliable_match",
                "matches": [], "message": _no_recent_message(duration_seconds),
                "live_debug": live_debug("best_match_too_old", len(within))}

    # Unique snapshot filenames (camera + search_id + frame_id) so the browser
    # never shows a cached older snapshot.
    fid = best["entry"]["frame_id"]
    frame_file = f"live_{camera_id}_{search_id}_f{fid}_frame.jpg"
    crop_file = f"live_{camera_id}_{search_id}_f{fid}_crop.jpg"
    entry = best["entry"]
    if entry.get("frame_jpeg"):
        frame = cv2.imdecode(np.frombuffer(entry["frame_jpeg"], np.uint8), cv2.IMREAD_COLOR)
        if frame is not None and best["bbox"]:
            x1, y1, x2, y2 = best["bbox"]
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                ok, cbuf = cv2.imencode(".jpg", crop)
                if ok:
                    _save(cbuf.tobytes(), crop_file)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ok, fbuf = cv2.imencode(".jpg", frame)
            if ok:
                _save(fbuf.tobytes(), frame_file)

    oldest_age = round(search_now - min(matched_ts), 1) if matched_ts else age_seconds
    newest_age = round(search_now - max(matched_ts), 1) if matched_ts else age_seconds
    best_sim = float(verdict["best_similarity"])
    match = {
        "track_id": 0,
        "decision": verdict["decision"],
        "best_similarity": best_sim,
        "avg_similarity": float(verdict["avg_similarity"]),
        "matched_frames": int(verdict["matched_frames"]),
        "match_timestamp": datetime.fromtimestamp(best_ts).strftime("%H:%M:%S"),
        "age_seconds": age_seconds,
        "timestamp_start": f"{oldest_age:.1f}s ago",
        "timestamp_end": f"{newest_age:.1f}s ago",
        "best_frame": f"/api/snapshot/{frame_file}",
        "best_crop": f"/api/snapshot/{crop_file}",
        "camera": camera_id, "camera_name": worker.name,
        "snapshot": frame_file, "confidence": round(best_sim * 100, 2),
        "time_formatted": f"{age_seconds:.1f}s ago",
    }
    return {
        "search_id": search_id, "status": "completed", "decision": "match_found",
        "matches": [match], "matches_by_camera": {camera_id: [match]},
        "live_debug": live_debug("matched_recent_frames", len(within)),
        "summary": {"total_matches": 1, "cameras_with_matches": 1, "best_similarity": best_sim},
    }
