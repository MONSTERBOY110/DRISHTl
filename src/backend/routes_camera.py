"""
Live-camera API routes.

Wires a phone / IP-webcam stream (e.g. http://PHONE_IP:8080/video) into Drishti:
connect & background-index, live MJPEG preview, status, disconnect, and a fast
search against the rolling embedding buffer. Static CCTVS search is unaffected.
"""

import asyncio
import logging
import shutil
import time
from datetime import datetime

from fastapi import APIRouter, Body, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.config.settings import Settings
from src.backend.camera.camera_manager import camera_manager
from src.backend.camera import live_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])
search_router = APIRouter(prefix="/api/search", tags=["camera"])

CONNECT_HELP = (
    "Could not connect to camera. Check that phone and laptop are on the same "
    "Wi-Fi, use http not https, and try the /video endpoint "
    "(e.g. http://PHONE_IP:8080/video)."
)


@router.post("/connect")
async def connect_camera(payload: dict = Body(...)):
    """Open a live stream, start background indexing, confirm it works."""
    camera_id = (payload.get("camera_id") or "cam01").strip()
    stream_url = (payload.get("stream_url") or "").strip()
    name = (payload.get("name") or "Phone Camera").strip()
    process_fps = payload.get("process_fps")
    buffer_seconds = payload.get("buffer_seconds")

    if not stream_url:
        return JSONResponse(status_code=400, content={
            "success": False, "message": "stream_url is required"})

    # Opening a network stream can block — run it off the event loop.
    result = await asyncio.to_thread(
        camera_manager.connect, camera_id, name, stream_url, process_fps, buffer_seconds)

    if not result.get("success"):
        logger.warning(f"camera connect failed [{camera_id}] {stream_url}: {result.get('error')}")
        return JSONResponse(status_code=502, content={
            "success": False,
            "message": CONNECT_HELP,
            "error": result.get("error"),
        })

    return {
        "success": True,
        "camera": result["camera"],
        "stream_path": f"/api/camera/stream/{camera_id}",
        "message": f"Connected to {name}. Indexing live frames…",
    }


@router.get("/status/{camera_id}")
async def camera_status(camera_id: str):
    info = camera_manager.status(camera_id)
    if info is None:
        return JSONResponse(status_code=404, content={
            "success": False, "message": f"Camera '{camera_id}' not connected"})
    return {"success": True, "status": info}


@router.get("/list")
async def list_cameras():
    return {"success": True, "cameras": camera_manager.list_cameras()}


@router.post("/disconnect/{camera_id}")
async def disconnect_camera(camera_id: str):
    ok = camera_manager.disconnect(camera_id)
    return {"success": ok, "message": "disconnected" if ok else "camera not found"}


@router.get("/stream/{camera_id}")
async def camera_stream(camera_id: str):
    """MJPEG live preview: <img src='/api/camera/stream/{camera_id}'>."""
    worker = camera_manager.get(camera_id)
    if worker is None:
        return JSONResponse(status_code=404, content={
            "success": False, "message": f"Camera '{camera_id}' not connected"})

    from src.backend.vision import config
    frame_interval = 1.0 / max(1.0, config.LIVE_PREVIEW_FPS)

    def generate():
        boundary = b"--frame"
        last_sent = None
        while True:
            w = camera_manager.get(camera_id)
            if w is None or w.status in ("disconnected", "error"):
                break
            jpeg = w.latest_jpeg()  # cached, downscaled frame from the single worker
            if jpeg is not None and jpeg is not last_sent:
                last_sent = jpeg
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            time.sleep(frame_interval)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@search_router.post("/live-buffer")
async def search_live_buffer(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    duration_seconds: int = Form(None),
):
    """Search only the last `duration_seconds` of an already-connected live camera."""
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"success": False, "message": "No file provided"})

        upload_path = Settings.UPLOAD_DIR / f"live_target_{datetime.now().timestamp()}.jpg"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        search_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(datetime.now().timestamp() * 1000) % 1000)
        results = await asyncio.to_thread(
            live_search.search_live_buffer, camera_id, str(upload_path), search_id, duration_seconds)

        return JSONResponse(status_code=200, content={
            "success": True,
            "search_id": search_id,
            "results": results,
            "message": results.get("message", "Live search completed"),
        })
    except Exception as e:
        logger.error(f"live-buffer search error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
