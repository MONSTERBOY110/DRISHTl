"""
Core v2 API (M1): health, cameras, index control, search, media.

Why one router here: TRD §4 gives M1 exactly these endpoints; every other
owner ships their own ``routes_*.py`` with its own APIRouter, and
``register_v2`` includes them all. Blocking work (indexing, search) runs in
``asyncio.to_thread`` like the baseline so the event loop never stalls.
Every response carries ``success`` (+ ``message`` on errors), as in v0.5.
"""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.backend.camera.camera_manager import camera_manager
from src.backend.vision import config as vision_config
from src.backend.vision import insightface_engine

from . import config, indexer, search as search_mod, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["v2"])


def _provider() -> str:
    """Provider in use once InsightFace has loaded; before that, the one it will pick."""
    return insightface_engine.active_provider() or insightface_engine._select_providers()[0]


def _index_summary() -> Dict[str, Any]:
    try:
        return store.get_store().summary()
    except Exception as e:  # health must never 500 because of the DB
        logger.warning(f"health: store summary failed: {e}")
        return {"cameras": 0, "tracks": 0, "last_indexed_at": None}


# ============================================================================
# 6.1 health
# ============================================================================

@router.get("/health")
async def health():
    return {
        "success": True,
        "version": config.VERSION,
        "index": _index_summary(),
        "models": {
            "face": vision_config.MODEL_PACK,
            "person": Path(config.PERSON_MODEL).stem,
            "appearance": f"{config.APPEARANCE_MODEL}/{config.APPEARANCE_PRETRAINED}",
        },
        "provider": _provider(),
    }


# ============================================================================
# 6.3 indexing
# ============================================================================

@router.post("/index/rebuild")
async def index_rebuild(payload: Dict[str, Any] = Body(default=None)):
    """Start an index run in the background; 409 while one is already running."""
    force = bool((payload or {}).get("force", False))
    if not indexer.start_background(force=force):
        return JSONResponse(status_code=409, content={
            "success": False, "state": "running", "message": "An index run is already in progress"})
    return {"success": True, "state": "running"}


@router.get("/index/status")
async def index_status():
    return {"success": True, **indexer.status()}


# ============================================================================
# 6.2 cameras: files in CCTVS/ (+ cameras.json) ∪ index state ∪ live workers
# ============================================================================

@router.get("/cameras")
async def cameras():
    st = store.get_store()
    db = {c["id"]: c for c in st.list_cameras()}
    counts = st.camera_track_counts()
    out: List[Dict[str, Any]] = []
    for cam in indexer.discover_cameras():
        row = db.get(cam["id"], {})
        has_map = cam.get("map_x") is not None and cam.get("map_y") is not None
        out.append({
            "id": cam["id"], "name": cam["name"], "location": cam.get("location"), "source": "file",
            "video_url": f"/api/cctv-video/{cam['file']}", "start_time": cam.get("start_time"),
            "map": {"x": cam["map_x"], "y": cam["map_y"]} if has_map else None,
            "indexed": bool(row.get("indexed_at")), "tracks": counts.get(cam["id"], 0),
        })
    live_cfg = {l.get("id"): l for l in indexer.load_camera_config().get("live", [])}
    for w in camera_manager.list_cameras():        # baseline phone/IP cameras, read-only
        spec = live_cfg.get(w["camera_id"], {})
        out.append({
            "id": w["camera_id"], "name": w.get("name") or w["camera_id"], "location": spec.get("location"),
            "source": "live", "video_url": f"/api/camera/stream/{w['camera_id']}", "start_time": None,
            "map": spec.get("map"), "indexed": False, "tracks": 0, "status": w.get("status"),
        })
    return {"success": True, "cameras": out}


# ============================================================================
# 6.4 search
# ============================================================================

@router.post("/search")
async def search_endpoint(
    photo: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None),
    cameras: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
):
    desc = (description or "").strip()
    has_photo = photo is not None and bool(photo.filename)
    if not has_photo and not desc:
        return JSONResponse(status_code=400, content={
            "success": False, "message": "Provide a photo, a description, or both."})

    photo_path: Optional[str] = None
    if has_photo:
        suffix = Path(photo.filename).suffix.lower() or ".jpg"
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            suffix = ".jpg"
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.UPLOADS_DIR / f"query_{int(time.time() * 1000)}{suffix}"
        with open(dest, "wb") as fh:
            shutil.copyfileobj(photo.file, fh)
        photo_path = str(dest)

    cam_ids = [c.strip() for c in cameras.split(",") if c.strip()] if cameras else None
    try:
        result = await asyncio.to_thread(search_mod.search, photo_path, desc or None, cam_ids)
    except Exception as e:
        logger.error(f"v2 search failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"Search error: {e}"})
    if case_id:
        result["case_id"] = case_id
    return result


# ============================================================================
# 6.8 media: bare filenames only, same traversal guard as the baseline /api/cctv-video
# ============================================================================

_MEDIA_DIRS = {"crops": config.CROPS_DIR, "frames": config.FRAMES_DIR, "cases": config.CASES_DIR}
_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@router.get("/media/{kind}/{filename}")
async def media(kind: str, filename: str):
    base_dir = _MEDIA_DIRS.get(kind)
    if base_dir is None:
        return JSONResponse(status_code=404, content={"success": False, "message": "Unknown media kind"})
    if (not filename or filename != os.path.basename(filename)
            or "/" in filename or "\\" in filename or filename in (".", "..")):
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid filename"})
    ext = Path(filename).suffix.lower()
    if ext not in _IMAGE_TYPES:
        return JSONResponse(status_code=400, content={"success": False, "message": "Unsupported file type"})
    base = base_dir.resolve()
    target = (base / filename).resolve()
    if target.parent != base:
        return JSONResponse(status_code=403, content={"success": False, "message": "Forbidden"})
    if not target.is_file():
        return JSONResponse(status_code=404, content={"success": False, "message": "Not found"})
    return FileResponse(target, media_type=_IMAGE_TYPES[ext], headers={"Cache-Control": "no-store"})
