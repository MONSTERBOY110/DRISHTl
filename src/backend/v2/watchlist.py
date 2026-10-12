"""
Watchlist daemon — live camera vs open cases (M4, task D4).  TRD §7.4.

Public surface
--------------
start()   Called once by register_v2() when this file exists.
stop()    Optional clean shutdown (e.g. tests).

How it works
------------
A single daemon thread wakes every WATCH_INTERVAL_S (1 s):

  for each live camera worker:
      entries = worker.buffer.recent(WATCH_WINDOW_S)   # last 10 s of frames
      for each open case with a face embedding:
          collect cosine similarities from all face detections in the window
          call decide_track_match(sims) using the SAME thresholds as search.py
          if strong_match or possible_match AND outside ALERT_COOLDOWN_S (60 s):
              save annotated frame + crop → alerts.publish(...)
              record cooldown so the same (case, camera) pair doesn't fire again
              for 60 s

The thread catches every exception so it can NEVER crash the server.
Embeddings in the rolling buffer are already L2-normalised (done by
camera_worker._index_frame).  So similarity = np.dot(emb_a, emb_b).

Read-only use of baseline objects (camera_manager, RollingBuffer).  No baseline
files are modified.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.backend.camera.camera_manager import camera_manager
from src.backend.vision import matcher as vision_matcher
from src.backend.v2 import alerts as alerts_mod
from src.backend.v2 import config
from src.backend.v2.cases import list_open_cases_with_face

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_started = False
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None

# Cooldown: {(case_id, camera_id): last_alert_epoch_s}
_cooldown: Dict[Tuple[str, str], float] = {}
_cooldown_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start() -> None:
    """Start the watchlist daemon thread.  Idempotent — safe to call multiple times."""
    global _started, _thread, _stop_event

    if _started:
        logger.info("watchlist: already running, skipping start()")
        return

    # Ensure the alert bus has the event loop captured before the first tick.
    alerts_mod.capture_event_loop()

    _stop_event.clear()
    _thread = threading.Thread(
        target=_run,
        name="v2-watchlist",
        daemon=True,
    )
    _thread.start()
    _started = True
    logger.info(f"watchlist: daemon started (interval={config.WATCH_INTERVAL_S}s, "
                f"window={config.WATCH_WINDOW_S}s, cooldown={config.ALERT_COOLDOWN_S}s)")


def stop() -> None:
    """Signal the watchlist thread to exit (used in tests / graceful shutdown)."""
    global _started
    _stop_event.set()
    _started = False
    logger.info("watchlist: stop requested")


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def _run() -> None:
    """Main loop — sleeps WATCH_INTERVAL_S between ticks."""
    while not _stop_event.is_set():
        time.sleep(config.WATCH_INTERVAL_S)
        try:
            _tick()
        except Exception:
            logger.exception("watchlist: unhandled error in _tick(); continuing")


def _tick() -> None:
    """One watchlist cycle: compare live cameras against open cases."""
    # 1. Get all open cases that have a face embedding -----------------------
    cases = list_open_cases_with_face()
    if not cases:
        return   # nothing to watch

    # 2. Get all connected live camera workers --------------------------------
    live = camera_manager.list_cameras()
    if not live:
        return   # no live cameras connected

    now = time.time()

    for cam_info in live:
        camera_id: str = cam_info["camera_id"]
        camera_name: str = cam_info.get("name") or camera_id
        location: Optional[str] = cam_info.get("location")

        worker = camera_manager.get(camera_id)
        if worker is None or worker.status != "connected":
            continue

        # Grab the recent window of processed frames
        entries, _, _ = worker.buffer.recent(config.WATCH_WINDOW_S)
        if not entries:
            continue

        for case in cases:
            case_id: str = case["id"]
            case_name: str = case["name"]
            q_emb: np.ndarray = case["face_emb"]   # (512,) float32, already L2-normed

            # Cooldown check -------------------------------------------------
            key = (case_id, camera_id)
            with _cooldown_lock:
                last = _cooldown.get(key, 0.0)
            if now - last < config.ALERT_COOLDOWN_S:
                continue

            # Collect similarities from all face detections in the window ----
            sims: List[float] = []
            best_sim = -1.0
            best_entry: Optional[Dict[str, Any]] = None
            best_face: Optional[Dict[str, Any]] = None

            for entry in entries:
                for face in entry.get("faces", []):
                    emb = face.get("embedding")
                    if emb is None:
                        continue
                    sim = float(np.dot(np.asarray(emb, dtype=np.float32), q_emb))
                    sims.append(sim)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
                        best_face = face

            if not sims:
                continue

            # Use the same decision logic as search.py (TRD §7.2) -----------
            verdict = vision_matcher.decide_track_match(sims)
            if verdict["decision"] == "no_reliable_match":
                continue

            # Match confirmed — fire alert -----------------------------------
            logger.info(
                f"watchlist: match! case={case_id} cam={camera_id} "
                f"decision={verdict['decision']} best_sim={best_sim:.3f} "
                f"frames={verdict['matched_frames']}"
            )

            snapshot_file, crop_file = _save_media(
                alert_id_hint=f"al_{int(now * 1000)}",
                entry=best_entry,
                face=best_face,
            )

            alerts_mod.publish(
                case_id=case_id,
                case_name=case_name,
                camera_id=camera_id,
                camera_name=camera_name,
                location=location,
                similarity=best_sim,
                decision=verdict["decision"],
                snapshot=snapshot_file,
                crop=crop_file,
            )

            # Record cooldown AFTER publish so a crash in publish doesn't
            # permanently silence this (case, camera) pair.
            with _cooldown_lock:
                _cooldown[key] = time.time()


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------

def _save_media(
    alert_id_hint: str,
    entry: Optional[Dict[str, Any]],
    face: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Decode the best entry's JPEG, draw bbox, save frame + crop.

    Returns (snapshot_filename, crop_filename) — bare filenames stored in
    data/v2/frames/ and data/v2/crops/ respectively.
    Returns (None, None) on any error so a media failure never blocks the alert.
    """
    if entry is None or not entry.get("frame_jpeg"):
        return None, None

    try:
        jpeg_bytes = entry["frame_jpeg"]
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None

        snapshot_name: Optional[str] = None
        crop_name: Optional[str] = None

        # Draw bounding box on a copy of the frame --------------------------
        annotated = frame.copy()
        if face is not None and "bbox" in face:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save annotated frame ----------------------------------------------
        snap_path = config.FRAMES_DIR / f"{alert_id_hint}.jpg"
        if cv2.imwrite(str(snap_path), annotated):
            snapshot_name = snap_path.name
        else:
            logger.warning(f"watchlist: could not write snapshot to {snap_path}")

        # Save face crop -----------------------------------------------------
        if face is not None and "bbox" in face:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop_img = frame[y1:y2, x1:x2]
                crop_path = config.CROPS_DIR / f"{alert_id_hint}.jpg"
                if cv2.imwrite(str(crop_path), crop_img):
                    crop_name = crop_path.name
                else:
                    logger.warning(f"watchlist: could not write crop to {crop_path}")

        return snapshot_name, crop_name

    except Exception:
        logger.exception("watchlist: _save_media failed")
        return None, None
