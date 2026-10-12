"""
Index-once pipeline (TRD §7.1): decode each camera's video ONCE into person tracks.

Why: the v0.5 pipeline re-decodes every video on every search (1-2 min on
CPU). Here each file is processed a single time into rows in the v2 store
(one per person track, with top-K face embeddings, an appearance embedding
and clothing colours); a search is then a few numpy dot products (< 1 s).

Per sampled frame: persons.PersonTracker (M2/B2) gives person boxes + track
ids; the baseline face detector + quality gate gives faces; associate (M2/B6)
pins faces to persons. Per track we keep the top-K crops (area x sharpness)
and top-K face embeddings (det_score x size); at the end appearance (M2/B3)
embeds the crops and names the clothing colours.

Skip rule: ``cameras.file_sig = "size:mtime:app=<appearance.MODEL_ID>"``. A
file is re-indexed only when the video changed or the appearance model
changed (so M2 swapping the stub for OpenCLIP re-indexes automatically
instead of mixing 64-D and 512-D rows). ``--force`` re-indexes everything.

Exactly one index run at a time; progress lives in ``PROGRESS`` (§6.3) and is
read by ``GET /api/v2/index/status``. CLI: ``python -m src.backend.v2.indexer [--force]``.
"""

import argparse
import heapq
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.backend.vision import detector, matcher, quality

from . import appearance, associate, config, persons, store

logger = logging.getLogger(__name__)

# Same allowlist as the baseline app.py (not imported from there: app.py imports us).
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

PROGRESS: Dict[str, Any] = {
    "state": "idle", "done": 0, "total": 0, "current_camera": None,
    "tracks_indexed": 0, "elapsed_s": 0.0, "error": None,
}
_run_lock = threading.Lock()
_started_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def load_camera_config(path: Path = None) -> Dict[str, Any]:
    """CCTVS/cameras.json (template docs/cameras.example.json) or {} when absent/invalid."""
    path = Path(path or config.CAMERAS_JSON)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"cameras.json unreadable ({e}); falling back to file names")
        return {}


def discover_cameras() -> List[Dict[str, Any]]:
    """Camera dicts (store schema keys) for every supported video in CCTVS/, config-enriched."""
    cfg = load_camera_config()
    by_file = {c.get("file"): c for c in cfg.get("cameras", []) if c.get("file")}
    video_dir = config.VIDEO_DIR
    if not video_dir.is_dir():
        return []
    files = sorted((p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS),
                   key=lambda p: _natural_key(p.name))
    cams: List[Dict[str, Any]] = []
    for idx, p in enumerate(files, start=1):
        spec = by_file.get(p.name, {})
        m = spec.get("map") or {}
        cams.append({
            "id": spec.get("id") or p.stem,
            "name": spec.get("name") or f"CCTV {idx}",
            "location": spec.get("location"),
            "source": "file",
            "file": p.name,
            "start_time": spec.get("start_time"),
            "map_x": m.get("x"), "map_y": m.get("y"),
        })
    missing = [c["file"] for c in by_file.values() if not (video_dir / c["file"]).is_file()]
    if missing:
        logger.warning(f"cameras.json lists files not in CCTVS/: {missing}")
    return cams


def file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}:app={appearance.MODEL_ID}"


# ---------------------------------------------------------------------------
# Per-track accumulator
# ---------------------------------------------------------------------------

@dataclass
class _TrackAcc:
    key: int
    t_start: float
    t_end: float
    n_frames: int = 0
    crops: List[Tuple[float, int, np.ndarray]] = field(default_factory=list)   # heap of (score, seq, crop)
    faces: List[Tuple[float, int, np.ndarray]] = field(default_factory=list)   # heap of (score, seq, emb)
    best_crop_score: float = -1.0
    best_crop: Optional[np.ndarray] = None
    best_frame_jpg: Optional[bytes] = None
    best_face_q: float = 0.0
    _seq: int = 0

    def _push(self, heap, k, score, payload):
        self._seq += 1
        item = (score, self._seq, payload)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)


def _clip_box(box, w, h):
    x1, y1, x2, y2 = (int(v) for v in box)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_camera(cam: Dict[str, Any], force: bool = False, st: Optional[store.Store] = None) -> Dict[str, Any]:
    """Index one file camera. Returns {camera_id, file, skipped, tracks, frames, seconds}."""
    st = st or store.get_store()
    path = config.VIDEO_DIR / cam["file"]
    sig = file_sig(path)
    existing = st.get_camera(cam["id"])
    st.upsert_camera(cam)  # keep name/location/map/start_time current even when we skip

    if not force and existing and existing.get("indexed_at") and existing.get("file_sig") == sig:
        n = st.camera_track_counts().get(cam["id"], 0)
        logger.info(f"[index] {cam['id']}: unchanged ({cam['file']}), {n} tracks kept")
        return {"camera_id": cam["id"], "file": cam["file"], "skipped": True, "tracks": n, "frames": 0, "seconds": 0.0}

    t0 = time.time()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = float(n_total / fps) if fps > 0 else None

    tracker = persons.PersonTracker()
    accs: Dict[int, _TrackAcc] = {}
    frame_pos = processed = 0
    faces_seen = faces_valid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_pos += 1
        if (frame_pos - 1) % config.FRAME_SKIP:
            continue
        processed += 1
        t = frame_pos / fps if fps > 0 else 0.0
        h, w = frame.shape[:2]

        dets = tracker.update(frame)
        if not dets:
            continue

        faces = detector.extract_faces(frame)
        faces_seen += len(faces)
        valid = [f for f in faces if f.get("embedding") is not None and quality.is_valid_face(f, frame)[0]]
        faces_valid += len(valid)
        assigned = associate.assign_faces(valid, dets) if valid else {}

        for track_id, box, _conf in dets:
            x1, y1, x2, y2 = _clip_box(box, w, h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            acc = accs.get(track_id)
            if acc is None:
                acc = accs[track_id] = _TrackAcc(key=track_id, t_start=t, t_end=t)
            acc.t_end = t
            acc.n_frames += 1

            crop = frame[y1:y2, x1:x2]
            sharp = quality.compute_blur_score(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            score = float((x2 - x1) * (y2 - y1)) * (sharp + 1.0)
            acc._push(acc.crops, config.TOPK_CROPS, score, crop.copy())
            if score > acc.best_crop_score:
                acc.best_crop_score = score
                acc.best_crop = crop.copy()
                annotated = frame.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                acc.best_frame_jpg = buf.tobytes() if ok else None

            face = assigned.get(track_id)
            if face is not None:
                emb = matcher.normalize(face["embedding"])
                if emb is not None:
                    fx1, fy1, fx2, fy2 = face["bbox"]
                    fq = float(face.get("det_score", 0.0)) * float(np.sqrt(max(1, (fx2 - fx1) * (fy2 - fy1))))
                    acc._push(acc.faces, config.TOPK_FACES, fq, emb)
                    acc.best_face_q = max(acc.best_face_q, fq)
    cap.release()

    # ---- persist ----------------------------------------------------------
    config.CROPS_DIR.mkdir(parents=True, exist_ok=True)
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    st.delete_tracks(cam["id"])
    n_tracks = 0
    for acc in accs.values():
        if acc.n_frames < config.MIN_TRACK_FRAMES or acc.best_crop is None:
            continue
        crops = [c for _, _, c in acc.crops]
        embs = appearance.embed_images(crops)
        app_emb = matcher.normalize(embs.mean(axis=0)) if len(embs) else None
        if app_emb is None:
            app_emb = np.zeros(appearance.DIM, dtype=np.float32)
        colours = appearance.color_attributes(acc.best_crop)
        name = f"{cam['id']}_t{acc.key}.jpg"
        cv2.imwrite(str(config.CROPS_DIR / name), acc.best_crop)
        if acc.best_frame_jpg:
            (config.FRAMES_DIR / name).write_bytes(acc.best_frame_jpg)
        face_embs = np.stack([e for _, _, e in acc.faces]).astype(np.float32) if acc.faces else None
        st.insert_track({
            "camera_id": cam["id"], "track_key": acc.key,
            "t_start": round(acc.t_start, 3), "t_end": round(acc.t_end, 3), "n_frames": acc.n_frames,
            "face_embs": face_embs, "face_count": 0 if face_embs is None else len(face_embs),
            "best_face_q": round(acc.best_face_q, 4) if face_embs is not None else None,
            "app_emb": app_emb, "upper_color": colours.get("upper"), "lower_color": colours.get("lower"),
            "best_crop": name, "best_frame": name if acc.best_frame_jpg else None,
        })
        n_tracks += 1
        PROGRESS["tracks_indexed"] += 1

    st.upsert_camera({"id": cam["id"], "fps": float(fps), "duration_s": duration_s,
                      "file_sig": sig, "indexed_at": store.now_iso()})
    st.invalidate()
    seconds = time.time() - t0
    logger.info(f"[index] {cam['id']}: {processed} frames, {len(accs)} raw tracks -> {n_tracks} kept, "
                f"faces {faces_valid}/{faces_seen} valid, {seconds:.1f}s")
    return {"camera_id": cam["id"], "file": cam["file"], "skipped": False, "tracks": n_tracks,
            "frames": processed, "seconds": round(seconds, 1)}


def _index_all_locked(force: bool, st: Optional[store.Store]) -> List[Dict[str, Any]]:
    global _started_at
    st = st or store.get_store()
    cams = discover_cameras()
    _started_at = time.time()
    PROGRESS.update({"state": "running", "done": 0, "total": len(cams), "current_camera": None,
                     "tracks_indexed": 0, "elapsed_s": 0.0, "error": None})
    results: List[Dict[str, Any]] = []
    try:
        for cam in cams:
            PROGRESS["current_camera"] = cam["id"]
            results.append(index_camera(cam, force=force, st=st))
            PROGRESS["done"] += 1
            PROGRESS["elapsed_s"] = round(time.time() - _started_at, 1)
        PROGRESS["state"] = "done"
    except Exception as e:
        logger.error(f"[index] failed on {PROGRESS.get('current_camera')}: {e}", exc_info=True)
        PROGRESS["state"] = "error"
        PROGRESS["error"] = str(e)
        raise
    finally:
        PROGRESS["current_camera"] = None
        PROGRESS["elapsed_s"] = round(time.time() - _started_at, 1)
    return results


def index_all(force: bool = False, st: Optional[store.Store] = None) -> List[Dict[str, Any]]:
    """Index every camera in CCTVS/ (blocking). Raises RuntimeError if a run is already active."""
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("index already running")
    try:
        return _index_all_locked(force, st)
    finally:
        _run_lock.release()


def start_background(force: bool = False) -> bool:
    """Start index_all in a daemon thread. False (and nothing started) if one is running."""
    if not _run_lock.acquire(blocking=False):
        return False

    def _run():
        try:
            _index_all_locked(force, None)
        except Exception:
            pass  # already logged and reflected in PROGRESS
        finally:
            _run_lock.release()

    threading.Thread(target=_run, name="v2-indexer", daemon=True).start()
    return True


def status() -> Dict[str, Any]:
    """Snapshot for GET /api/v2/index/status (§6.3)."""
    snap = dict(PROGRESS)
    if snap["state"] == "running" and _started_at:
        snap["elapsed_s"] = round(time.time() - _started_at, 1)
    return snap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Index every video in CCTVS/ into the v2 store.")
    ap.add_argument("--force", action="store_true", help="re-index even when a file is unchanged")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    st = store.get_store()
    st.init_db()
    t0 = time.time()
    results = index_all(force=args.force, st=st)
    print(f"\n{'camera':<12} {'file':<28} {'tracks':>6}  {'status':<8} {'secs':>6}")
    for r in results:
        print(f"{r['camera_id']:<12} {r['file']:<28} {r['tracks']:>6}  "
              f"{'skipped' if r['skipped'] else 'indexed':<8} {r['seconds']:>6}")
    print(f"\n{len(results)} camera(s), {st.count_tracks()} tracks in store, {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
