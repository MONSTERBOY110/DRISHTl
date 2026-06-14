"""
Open-set face-search pipeline (InsightFace backend).

Flow per search:
    target image -> detect + quality gate + embed (reject if no clean face)
    each video   -> per frame: detect -> quality gate -> embed -> cosine vs target
                                -> IoU track association -> accumulate per track
    per track    -> open-set decision (strong / possible / no reliable)
    overall      -> match_found if any qualifying track, else no_reliable_match

Returns results purely in memory; snapshot/crop writes are best-effort and never
affect the returned decision.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config.settings import Settings

from . import config, detector, matcher, quality, recognizer
from .tracker import IoUTracker

logger = logging.getLogger(__name__)

NO_MATCH_MESSAGE = "No reliable match found in the selected footage."


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_ms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def _no_reliable_match(search_id: str, message: str = NO_MATCH_MESSAGE,
                       rejected: Dict[str, int] = None,
                       stats: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "search_id": search_id,
        "status": "completed",
        "decision": "no_reliable_match",
        "matches": [],
        "message": message,
        "rejected": rejected or quality.new_reject_counter(),
        "stats": stats or {"matches_found": 0},
    }


def _select_target_face(faces: List[Dict[str, Any]], frame_bgr: np.ndarray,
                        rejected: Dict[str, int]):
    """Pick the largest valid face from the target photo, or None."""
    valid = []
    for f in faces:
        ok, reason = quality.is_valid_face(
            f, frame_bgr,
            min_det=config.TARGET_MIN_DETECTION_SCORE,
            min_size=config.TARGET_MIN_FACE_SIZE,
            min_blur=config.TARGET_MIN_BLUR_SCORE,
        )
        if ok:
            valid.append(f)
        elif reason:
            rejected[reason] += 1
    if not valid:
        return None
    # Largest bbox area = most prominent subject.
    return max(valid, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))


def _save_image(img: np.ndarray, filename: str) -> bool:
    try:
        Settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(Settings.RESULTS_DIR / filename), img))
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(f"Could not save image {filename}: {e}")
        return False


def search(target_path: str, video_paths: List[Path], search_id: str) -> Dict[str, Any]:
    """Run the open-set face search. Returns the full result dict."""
    rejected = quality.new_reject_counter()
    stats: Dict[str, Any] = {
        "backend": config.RECOGNIZER_BACKEND,
        "total_videos": len(video_paths),
        "videos_processed": 0,
        "total_frames_processed": 0,
        "faces_detected": 0,
        "faces_rejected": 0,
        "matches_found": 0,
    }

    # ---- Target image -----------------------------------------------------
    target_img = cv2.imread(target_path)
    if target_img is None:
        return {
            "search_id": search_id, "status": "error",
            "decision": "no_reliable_match", "error": "Could not read uploaded image",
            "matches": [], "rejected": rejected, "stats": stats,
        }

    target_faces = detector.extract_faces(target_img)
    if not target_faces:
        return _no_reliable_match(
            search_id, "No face detected in the uploaded photo.", rejected, stats)

    best_target = _select_target_face(target_faces, target_img, rejected)
    if best_target is None:
        return _no_reliable_match(
            search_id,
            "The face in the uploaded photo is too low quality to search reliably.",
            rejected, stats)

    target_embedding = matcher.normalize(recognizer.get_embedding(best_target, target_img))
    if target_embedding is None:
        return _no_reliable_match(
            search_id, "Could not extract a usable face embedding from the photo.",
            rejected, stats)

    logger.info(f"[{search_id}] Target face embedded. Searching {len(video_paths)} video(s).")

    # ---- Helpers to persist snapshots + build a match record -------------
    def _save_track_images(track, camera):
        frame_file = f"{search_id}_{camera}_track{track.track_id}_frame.jpg"
        crop_file = f"{search_id}_{camera}_track{track.track_id}_crop.jpg"
        if track.best_frame_img is not None:
            annotated = track.best_frame_img.copy()
            bx1, by1, bx2, by2 = track.best_bbox
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            _save_image(annotated, frame_file)
        if track.best_crop_img is not None:
            _save_image(track.best_crop_img, crop_file)
        return frame_file, crop_file

    def _record(track, verdict, camera, camera_name):
        frame_file, crop_file = _save_track_images(track, camera)
        best_sim = float(verdict["best_similarity"])
        return {
            # ---- new open-set fields ----
            "track_id": track.track_id,
            "decision": verdict["decision"],
            "timestamp_start": _format_hms(track.ts_start),
            "timestamp_end": _format_hms(track.ts_end),
            "best_frame": f"/api/snapshot/{frame_file}",
            "best_crop": f"/api/snapshot/{crop_file}",
            "avg_similarity": float(verdict["avg_similarity"]),
            "best_similarity": best_sim,
            "matched_frames": int(verdict["matched_frames"]),
            # ---- backward-compat fields (frontend) ----
            "camera": camera,
            "camera_name": camera_name,
            "snapshot": frame_file,
            "timestamp": round(track.ts_end, 2),
            "frame_number": track.best_frame_number,
            "time_formatted": _format_ms(track.ts_start),
            "confidence": round(best_sim * 100, 2),
        }

    # ---- Videos -----------------------------------------------------------
    matches: List[Dict[str, Any]] = []                          # flat (compat)
    matches_by_camera: Dict[str, List[Dict[str, Any]]] = {}
    per_camera_debug: Dict[str, Dict[str, Any]] = {}
    near_misses: List[Dict[str, Any]] = []

    for cam_index, video_path in enumerate(video_paths, start=1):
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            continue

        camera = video_path.stem
        camera_name = f"CCTV {cam_index}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tracker = IoUTracker()
        frame_pos = 0
        processed = 0
        cam_rejected = quality.new_reject_counter()
        cam_faces_detected = 0
        cam_valid = 0
        best_seen = -1.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_pos += 1
            if (frame_pos - 1) % config.FRAME_SKIP != 0:
                continue
            processed += 1

            tracker.begin_frame()
            faces = detector.extract_faces(frame)
            cam_faces_detected += len(faces)
            timestamp = frame_pos / fps if fps > 0 else 0.0

            for face in faces:
                ok, reason = quality.is_valid_face(face, frame)
                if not ok:
                    if reason:
                        cam_rejected[reason] += 1
                    continue

                cam_valid += 1
                emb = matcher.normalize(recognizer.get_embedding(face, frame))
                if emb is None:
                    continue
                sim = matcher.cosine(target_embedding, emb)
                if sim > best_seen:
                    best_seen = sim

                x1, y1, x2, y2 = face["bbox"]
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                crop_copy = crop.copy() if crop.size > 0 else None

                track = tracker.assign(face["bbox"])
                track.update(face["bbox"], sim, timestamp, frame, crop_copy, frame_pos)

        cap.release()

        # ---- Track-level decision for this camera ----
        cam_matches: List[Dict[str, Any]] = []
        tracks_strong = tracks_possible = 0
        best_track = None
        for track in tracker.tracks:
            if best_track is None or track.best_similarity > best_track.best_similarity:
                best_track = track
            verdict = matcher.decide_track_match(track.similarities)
            if verdict["decision"] == "no_reliable_match":
                continue
            if verdict["decision"] == "strong_match":
                tracks_strong += 1
            else:
                tracks_possible += 1
            cam_matches.append(_record(track, verdict, camera, camera_name))

        # Sort this camera's tracks: strong first, then similarity, then frames.
        cam_matches.sort(
            key=lambda m: (m["decision"] != "strong_match", -m["best_similarity"], -m["matched_frames"])
        )
        returned = cam_matches[: config.MAX_RESULTS_PER_CAMERA]
        is_near_miss = (not cam_matches) and best_track is not None and \
            best_seen >= (config.POSSIBLE_MATCH_THRESHOLD - config.NEAR_MISS_MARGIN)

        # Accumulate global (backward-compat) stats + rejection counters.
        for k in cam_rejected:
            rejected[k] += cam_rejected[k]
        stats["faces_detected"] += cam_faces_detected
        stats["faces_rejected"] += sum(cam_rejected.values())
        stats["videos_processed"] += 1
        stats["total_frames_processed"] += processed

        if cam_matches:
            best_decision = returned[0]["decision"]
            drop_reason = None
        elif is_near_miss:
            best_decision = "near_miss"
            drop_reason = "below_threshold_or_insufficient_frames"
        elif best_seen <= 0:
            best_decision = "no_face"
            drop_reason = "no_valid_face_for_target"
        else:
            best_decision = "no_match"
            drop_reason = "below_threshold"

        per_camera_debug[camera] = {
            "camera_name": camera_name,
            "frames_processed": processed,
            "faces_detected": cam_faces_detected,
            "faces_rejected": sum(cam_rejected.values()),
            "rejected": cam_rejected,
            "valid_faces": cam_valid,
            "tracks_total": len(tracker.tracks),
            "tracks_strong": tracks_strong,
            "tracks_possible": tracks_possible,
            "qualifying_tracks": len(cam_matches),
            "returned_tracks": len(returned),
            "best_similarity_seen": round(best_seen, 4) if best_seen >= 0 else 0.0,
            "best_track_frames": max((m["matched_frames"] for m in cam_matches), default=0),
            "best_decision": best_decision,
            "drop_reason": drop_reason,
            "decision": "match_found" if cam_matches else "no_reliable_match",
        }

        if cam_matches:
            matches_by_camera[camera] = returned
            matches.extend(cam_matches)
        elif is_near_miss:
            frame_file, crop_file = _save_track_images(best_track, camera)
            near_misses.append({
                "camera": camera,
                "camera_name": camera_name,
                "best_similarity": round(best_seen, 4),
                "matched_frames": len(best_track.similarities),
                "reason": "below threshold or insufficient matched frames",
                "best_frame": f"/api/snapshot/{frame_file}",
                "best_crop": f"/api/snapshot/{crop_file}",
            })

        logger.info(
            f"[{search_id}] {camera}: {processed} frames, "
            f"{len(cam_matches)} qualifying track(s), best_sim={best_seen:.3f}"
        )

    # ---- Aggregate --------------------------------------------------------
    if config.RESULT_MODE == "per_camera":
        # One block per camera (already capped at MAX_RESULTS_PER_CAMERA), in
        # camera order — every camera with a qualifying match is represented.
        final_matches: List[Dict[str, Any]] = []
        for cam in matches_by_camera:
            final_matches.extend(matches_by_camera[cam])
    else:  # "global" (legacy)
        final_matches = sorted(
            matches, key=lambda m: (m["decision"] != "strong_match", -m["best_similarity"])
        )[: config.MAX_RESULTS]

    stats["matches_found"] = len(final_matches)

    # Coverage summary across every camera that was processed.
    all_camera_ids = list(per_camera_debug.keys())
    matched_camera_ids = list(matches_by_camera.keys())
    missing_camera_ids = [c for c in all_camera_ids if c not in matches_by_camera]
    camera_coverage = {
        "total_cameras": len(all_camera_ids),
        "cameras_with_matches": len(matched_camera_ids),
        "cameras_without_matches": len(missing_camera_ids),
        "matched_camera_ids": matched_camera_ids,
        "missing_camera_ids": missing_camera_ids,
    }

    if not final_matches:
        result = _no_reliable_match(search_id, NO_MATCH_MESSAGE, rejected, stats)
        result["matches_by_camera"] = {}
        result["near_misses"] = near_misses
        result["per_camera_debug"] = per_camera_debug
        result["camera_coverage"] = camera_coverage
        return result

    return {
        "search_id": search_id,
        "status": "completed",
        "decision": "match_found",
        "matches": final_matches,
        "matches_by_camera": matches_by_camera,
        "camera_coverage": camera_coverage,
        "near_misses": near_misses,
        "per_camera_debug": per_camera_debug,
        "rejected": rejected,
        "stats": stats,
        "summary": {
            "total_matches": len(final_matches),
            "cameras_with_matches": len(matched_camera_ids),
            "strong_matches": sum(1 for m in final_matches if m["decision"] == "strong_match"),
            "possible_matches": sum(1 for m in final_matches if m["decision"] == "possible_match"),
            "best_similarity": max((m["best_similarity"] for m in final_matches), default=0.0),
        },
    }
