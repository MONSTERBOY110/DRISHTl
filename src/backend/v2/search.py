"""
Search fusion (TRD §7.2-7.3): photo and/or description → verified sightings,
unverified leads, and a cross-camera trail. Returns the exact §6.4 shape.

Why two separate lists: a *verified* sighting means the baseline open-set face
decision (vision.matcher.decide_track_match with the calibrated thresholds in
vision/config.py) said strong/possible match. A *lead* is appearance-only
(text↔image similarity z-score plus clothing-colour bonuses) and is never a
match claim; the UI labels it "Lead (unverified)". No fake confidence %.

Face thresholds are reused, never re-invented. The photo goes through the same
lenient target gate as v0.5 (TARGET_MIN_* via pipeline._select_target_face).
Scoring constants (ATTR_BONUS_Z, LEAD_MIN_Z, TRAIL_MIN_Z, LEADS_MAX) are M2's
to calibrate in B5. With M1's stub appearance model embed_text() is zero, so
leads come only from colour attributes until M2's OpenCLIP lands.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.backend.camera.camera_manager import camera_manager
from src.backend.vision import detector, matcher, quality
from src.backend.vision.pipeline import _select_target_face

from . import appearance, config, indexer, persons, query_parser, store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query side
# ---------------------------------------------------------------------------

def embed_photo(photo_path: str) -> Tuple[Optional[np.ndarray], str]:
    """Normalised ArcFace embedding of the most prominent valid face, or (None, reason)."""
    img = cv2.imread(str(photo_path))
    if img is None:
        return None, "unreadable"
    faces = detector.extract_faces(img)
    if not faces:
        return None, "no_face"
    best = _select_target_face(faces, img, quality.new_reject_counter())
    if best is None:
        return None, "low_quality"
    emb = matcher.normalize(best.get("embedding"))
    if emb is None:
        return None, "no_embedding"
    return emb.astype(np.float32), "ok"


def _hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _wall_time(cam: Optional[Dict[str, Any]], t: float) -> Optional[str]:
    """camera.start_time + t seconds as ISO-8601, or None when the camera has no clock."""
    start = (cam or {}).get("start_time")
    if not start:
        return None
    try:
        return (datetime.fromisoformat(start) + timedelta(seconds=float(t))).isoformat(timespec="seconds")
    except ValueError:
        return None


def _requested_colour_slots(parsed: Dict[str, Any]) -> int:
    return int(bool(parsed.get("upper"))) + int(bool(parsed.get("lower")))


def _lead_floor(parsed: Dict[str, Any]) -> float:
    """Let one explicit clothing colour produce an unverified lead."""
    if _requested_colour_slots(parsed) <= 1:
        return min(config.LEAD_MIN_Z, config.ATTR_BONUS_Z)
    return config.LEAD_MIN_Z


def _trail_floor(parsed: Dict[str, Any]) -> float:
    if _requested_colour_slots(parsed) <= 1:
        return min(config.TRAIL_MIN_Z, _lead_floor(parsed))
    return config.TRAIL_MIN_Z


def _attribute_score(parsed: Dict[str, Any], upper: Optional[str], lower: Optional[str]) -> Tuple[float, List[str]]:
    score = 0.0
    hits: List[str] = []
    
    # Matching upper color awards bonus; explicit mismatch with a recognized color penalizes
    if parsed.get("upper"):
        if upper in parsed["upper"]:
            score += config.ATTR_BONUS_Z
            hits.append(f"upper:{upper}")
        elif upper is not None:
            score -= 1.5
            
    # Matching lower color awards bonus; explicit mismatch with a recognized color penalizes
    if parsed.get("lower"):
        if lower in parsed["lower"]:
            score += config.ATTR_BONUS_Z
            hits.append(f"lower:{lower}")
        elif lower is not None:
            score -= 1.5

    return score, hits


def _live_specs() -> Dict[str, Dict[str, Any]]:
    cfg = indexer.load_camera_config()
    return {str(l.get("id")): l for l in cfg.get("live", []) if l.get("id")}


def _decode_live_frame(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    raw = entry.get("frame_jpeg") or b""
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _live_description_sightings(parsed: Dict[str, Any], camera_ids: Optional[Sequence[str]], q_face: Optional[np.ndarray] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Appearance-only leads from live rolling frames, with optional face matching."""
    wanted = set(camera_ids or [])
    specs = _live_specs()
    workers = []
    for status in camera_manager.list_cameras():
        cid = status["camera_id"]
        if wanted and cid not in wanted:
            continue
        worker = camera_manager.get(cid)
        if worker is not None:
            workers.append((status, worker))
    if not workers:
        return [], 0

    text_vec = np.asarray(appearance.embed_text(parsed["clip_text"]), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for status, worker in workers:
        entries, _total, _dropped = worker.buffer.recent(config.LIVE_DESC_WINDOW_S)
        if config.LIVE_DESC_MAX_FRAMES > 0:
            entries = entries[-config.LIVE_DESC_MAX_FRAMES:]
        tracker = persons.PersonTracker()
        for entry in entries:
            frame = _decode_live_frame(entry)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            dets = tracker.update(frame)
            if not dets:
                dets = [(int(entry.get("frame_id") or 0), (0, 0, w, h), 0.25)]
            for track_key, box, _conf in dets:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = frame[y1:y2, x1:x2]
                app = appearance.embed_images([crop])
                score = float(app[0] @ text_vec) if len(app) else 0.0
                colours = appearance.color_attributes(crop)
                rows.append({
                    "camera_id": status["camera_id"],
                    "camera_name": status.get("name") or status["camera_id"],
                    "location": specs.get(status["camera_id"], {}).get("location"),
                    "map": specs.get(status["camera_id"], {}).get("map"),
                    "track_key": int(track_key),
                    "timestamp": float(entry.get("timestamp") or time.time()),
                    "frame_id": int(entry.get("frame_id") or 0),
                    "score": score,
                    "upper": colours.get("upper"),
                    "lower": colours.get("lower"),
                    "crop": crop,
                    "frame": frame,
                })
    if not rows:
        return [], 0

    scores = np.array([r["score"] for r in rows], dtype=np.float32)
    std = float(scores.std())
    base_z = (scores - scores.mean()) / std if std > 1e-6 else np.zeros_like(scores)
    sightings: List[Dict[str, Any]] = []
    floor = _lead_floor(parsed)
    for i in np.argsort(-base_z):
        r = rows[int(i)]
        bonus, hits = _attribute_score(parsed, r["upper"], r["lower"])
        z = float(base_z[int(i)] + bonus)
        if z < floor:
            continue

        # Save live crop and frame so the UI can display actual sighting thumbnails
        crop_fname = f"live_{r['camera_id']}_{r['frame_id']}_{r['track_key']}.jpg"
        frame_fname = f"live_frame_{r['camera_id']}_{r['frame_id']}_{r['track_key']}.jpg"
        crop_path = config.CROPS_DIR / crop_fname
        frame_path = config.FRAMES_DIR / frame_fname
        if r.get("crop") is not None and not crop_path.exists():
            cv2.imwrite(str(crop_path), r["crop"])
        if r.get("frame") is not None and not frame_path.exists():
            cv2.imwrite(str(frame_path), r["frame"])

        # Check face match if photo face embedding is available
        face_info = None
        evidence = "appearance"
        if q_face is not None and r.get("crop") is not None:
            try:
                faces = detector.extract_faces(r["crop"])
                if faces:
                    best_sim = -1.0
                    for f in faces:
                        f_emb = matcher.normalize(f.get("embedding"))
                        if f_emb is not None:
                            sim = float(f_emb @ q_face)
                            if sim > best_sim:
                                best_sim = sim
                    if best_sim >= 0.40:
                        dec = "strong_match" if best_sim >= 0.50 else "possible_match"
                        face_info = {"decision": dec, "best_similarity": round(best_sim, 2), "matched_frames": 1}
                        evidence = "face+appearance"
            except Exception:
                pass

        wall = datetime.fromtimestamp(r["timestamp"]).astimezone().isoformat(timespec="seconds")
        sightings.append({
            "sighting_id": f"live_{r['camera_id']}_{r['frame_id']}_{r['track_key']}",
            "track_id": r["track_key"],
            "camera_id": r["camera_id"],
            "camera_name": r["camera_name"],
            "location": r["location"],
            "t_start": 0.0,
            "t_end": 0.0,
            "t_start_hms": "LIVE",
            "wall_time": wall,
            "face": face_info,
            "appearance": {"score": round(r["score"], 4), "z": round(z, 3),
                           "rank": None, "attribute_hits": hits},
            "evidence": evidence,
            "best_crop": f"/api/v2/media/crops/{crop_fname}" if crop_path.exists() else None,
            "best_frame": f"/api/v2/media/frames/{frame_fname}" if frame_path.exists() else None,
            "clip_url": f"/api/camera/stream/{r['camera_id']}",
        })
        if len(sightings) >= config.LEADS_MAX:
            break
    for rank, sighting in enumerate(sightings, start=1):
        sighting["appearance"]["rank"] = rank
    return sightings, len(rows)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(photo_path: Optional[str] = None, description: Optional[str] = None,
           camera_ids: Optional[Sequence[str]] = None, st: Optional[store.Store] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    st = st or store.get_store()
    cams = {c["id"]: c for c in st.list_cameras()}
    live_specs = _live_specs()
    for status in camera_manager.list_cameras():
        spec = live_specs.get(status["camera_id"], {})
        m = spec.get("map") or {}
        cams.setdefault(status["camera_id"], {
            "id": status["camera_id"],
            "name": status.get("name") or status["camera_id"],
            "location": spec.get("location"),
            "source": "live",
            "map_x": m.get("x"),
            "map_y": m.get("y"),
        })
    cam_filter = [c for c in camera_ids if c] if camera_ids else None
    mats = st.get_matrices(appearance.DIM, cam_filter or None)
    n = len(mats["camera_ids"])

    # ---- query: photo ------------------------------------------------------
    has_photo = bool(photo_path)
    q_face: Optional[np.ndarray] = None
    photo_reason = None
    if has_photo:
        q_face, photo_reason = embed_photo(photo_path)
    photo_face_found = q_face is not None

    # ---- query: description -----------------------------------------------
    desc = (description or "").strip() or None
    parsed = query_parser.parse(desc) if desc else None
    lead_floor = _lead_floor(parsed) if parsed is not None else config.LEAD_MIN_Z
    trail_floor = _trail_floor(parsed) if parsed is not None else config.TRAIL_MIN_Z

    # ---- face verification (TRD §7.2) --------------------------------------
    verdicts: Dict[int, Dict[str, Any]] = {}      # row -> decide_track_match() result
    verified_rows: List[int] = []
    if q_face is not None:
        for i, fe in enumerate(mats["face_embs"]):
            if fe is None:
                continue
            v = matcher.decide_track_match((fe @ q_face).tolist())
            verdicts[i] = v
            if v["decision"] != "no_reliable_match":
                verified_rows.append(i)

    # ---- appearance leads --------------------------------------------------
    app_s = lead_z = None
    attr_hits: List[List[str]] = [[] for _ in range(n)]
    if parsed is not None and n:
        t_vec = np.asarray(appearance.embed_text(parsed["clip_text"]), dtype=np.float32)
        app_s = mats["app_embs"] @ t_vec
        std = float(app_s.std())
        z = (app_s - app_s.mean()) / std if std > 1e-6 else np.zeros_like(app_s)   # zero text vector → z = 0
        bonus = np.zeros(n, dtype=np.float32)
        for i in range(n):
            score, hits = _attribute_score(parsed, mats["upper"][i], mats["lower"][i])
            bonus[i] += score
            attr_hits[i].extend(hits)
        lead_z = z + bonus

    # ---- sightings ---------------------------------------------------------
    def face_block(i: int) -> Optional[Dict[str, Any]]:
        v = verdicts.get(i)
        if v is None:
            return None
        return {"decision": v["decision"], "best_similarity": float(v["best_similarity"]),
                "matched_frames": int(v["matched_frames"])}

    def app_block(i: int, rank: Optional[int]) -> Optional[Dict[str, Any]]:
        if lead_z is None:
            return None
        return {"score": round(float(app_s[i]), 4), "z": round(float(lead_z[i]), 3),
                "rank": rank, "attribute_hits": attr_hits[i]}

    def sighting(i: int, evidence: str, rank: Optional[int] = None) -> Dict[str, Any]:
        cam_id = mats["camera_ids"][i]
        cam = cams.get(cam_id, {})
        t_start, t_end = float(mats["t_start"][i]), float(mats["t_end"][i])
        track_id = int(mats["track_ids"][i])
        crop, frame = mats["best_crop"][i], mats["best_frame"][i]
        return {
            "sighting_id": f"t_{track_id}", "track_id": track_id,
            "camera_id": cam_id, "camera_name": cam.get("name") or cam_id, "location": cam.get("location"),
            "t_start": round(t_start, 2), "t_end": round(t_end, 2), "t_start_hms": _hms(t_start),
            "wall_time": _wall_time(cam, t_start),
            "face": face_block(i), "appearance": app_block(i, rank), "evidence": evidence,
            "best_crop": f"/api/v2/media/crops/{crop}" if crop else None,
            "best_frame": f"/api/v2/media/frames/{frame}" if frame else None,
            "clip_url": f"/api/cctv-video/{cam['file']}#t={t_start:.1f}" if cam.get("file") else None,
        }

    # When both photo and description are given:
    # First filter by description; across candidate tracks that match description,
    # match face and return the most accurate ones as verified.
    if q_face is not None and lead_z is not None:
        # Exclude tracks with explicit clothing color conflicts (lead_z < -0.2)
        verified_desc_rows = [i for i in verified_rows if lead_z[i] >= -0.2]
        verified = [sighting(i, "face+appearance") for i in verified_desc_rows]
        verified.sort(key=lambda s: (
            s["face"]["decision"] != "strong_match",
            -(float(s["appearance"]["z"]) * 0.4 + float(s["face"]["best_similarity"]) * 0.6) if s.get("appearance") else -float(s["face"]["best_similarity"]),
            -s["face"]["best_similarity"]
        ))
        taken = set(verified_desc_rows)
    else:
        verified = [sighting(i, "face") for i in verified_rows]
        verified.sort(key=lambda s: (s["face"]["decision"] != "strong_match",
                                     -s["face"]["best_similarity"], -s["face"]["matched_frames"]))
        taken = set(verified_rows)

    leads: List[Dict[str, Any]] = []
    if lead_z is not None:
        seen_per_cam: Dict[str, List[float]] = {}
        for i in np.argsort(-lead_z):
            i = int(i)
            if lead_z[i] < lead_floor:
                break
            if i in taken:
                continue
            cam_id, ts = mats["camera_ids"][i], float(mats["t_start"][i])
            # ByteTrack fragments: one lead per camera per LEAD_DEDUP_S window (TRD §13)
            if any(abs(ts - prev) < config.LEAD_DEDUP_S for prev in seen_per_cam.get(cam_id, [])):
                continue
            seen_per_cam.setdefault(cam_id, []).append(ts)
            leads.append(sighting(i, "appearance", rank=len(leads) + 1))
            if len(leads) >= config.LEADS_MAX:
                break

    live_scanned = 0
    if parsed is not None:
        live_leads, live_scanned = _live_description_sightings(parsed, cam_filter, q_face)
        for live in live_leads:
            if live.get("face") and live["face"]["best_similarity"] >= 0.45:
                verified.append(live)
            else:
                if len(leads) < config.LEADS_MAX:
                    live["appearance"]["rank"] = len(leads) + 1
                    leads.append(live)

    # ---- trail (TRD §7.3) ----------------------------------------------------
    trail_input = verified + [l for l in leads if l["appearance"]["z"] >= trail_floor]
    trail = build_trail(trail_input, cams)
    last_seen = trail[-1] if trail else None

    # ---- decision + message ------------------------------------------------
    if verified:
        decision = "match_found"
    elif leads:
        decision = "leads_only"
    else:
        decision = "no_reliable_match"
    msg: List[str] = []
    if n == 0 and live_scanned == 0:
        msg.append("Index is empty: run POST /api/v2/index/rebuild first.")
    if has_photo and not photo_face_found:
        msg.append({"unreadable": "Could not read the photo.", "no_face": "No face detected in the photo.",
                    "low_quality": "The face in the photo is too low quality to search.",
                    }.get(photo_reason, "Could not use the photo."))
    if verified:
        msg.append(f"Verified in {len({s['camera_id'] for s in verified})} camera(s)")
    elif q_face is not None:
        msg.append("No reliable face match")
    if parsed is not None:
        msg.append(f"{len(leads)} appearance lead(s), unverified")
    message = "; ".join(msg) + "." if msg else "Nothing to search."

    return {
        "success": True,
        "query": {
            "has_photo": has_photo, "photo_face_found": photo_face_found, "description": desc,
            "parsed": {k: (parsed or {}).get(k, [] if k not in ("age_hint", "clip_text") else None)
                       for k in ("upper", "lower", "keywords", "facial_features", "physical_features", "age_hint", "clip_text")},
        },
        "decision": decision,
        "verified": verified,
        "leads": leads,
        "trail": trail,
        "last_seen": last_seen,
        "timings": {"search_ms": int((time.perf_counter() - t0) * 1000), "tracks_scanned": n + live_scanned},
        "message": message,
    }


# ---------------------------------------------------------------------------
# Trail
# ---------------------------------------------------------------------------

def build_trail(sightings: List[Dict[str, Any]], cams: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """TrailPoints oldest → newest: sorted by wall_time (else camera order + t_start),
    consecutive same-camera points closer than TRAIL_MERGE_S merged, face evidence wins."""
    if not sightings:
        return []
    cam_order = {cid: k for k, cid in enumerate(cams)}
    points = []
    for s in sightings:
        cam = cams.get(s["camera_id"], {})
        has_map = cam.get("map_x") is not None and cam.get("map_y") is not None
        points.append({
            "camera_id": s["camera_id"], "camera_name": s["camera_name"], "location": s["location"],
            "map": {"x": cam["map_x"], "y": cam["map_y"]} if has_map else None,
            "t_start": s["t_start"], "t_end": s["t_end"], "wall_time": s["wall_time"],
            "evidence": s["evidence"], "sighting_ids": [s["sighting_id"]],
        })

    if all(p["wall_time"] for p in points):
        points.sort(key=lambda p: datetime.fromisoformat(p["wall_time"]))
    else:
        points.sort(key=lambda p: (cam_order.get(p["camera_id"], 10**6), p["t_start"]))

    merged: List[Dict[str, Any]] = []
    for p in points:
        last = merged[-1] if merged else None
        if last and last["camera_id"] == p["camera_id"] and p["t_start"] - last["t_end"] < config.TRAIL_MERGE_S:
            last["t_end"] = max(last["t_end"], p["t_end"])
            last["sighting_ids"].append(p["sighting_ids"][0])
            if p["evidence"] == "face":
                last["evidence"] = "face"
            continue
        merged.append(p)
    return merged
