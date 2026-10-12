"""
DRISTI - Lost Person Detection System
Main FastAPI application for face recognition in CCTV footage.
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
import sys
import asyncio

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.search_service import SearchService
from src.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Settings.LOG_DIR / 'drishti.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize directories
Settings.create_directories()

# ============================================================================
# CCTV VIDEO DISCOVERY (local CCTVS/ folder — no remote/Cloudinary URLs)
# ============================================================================
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
}


def _natural_key(name: str):
    """Sort key so cctv2 < cctv10 (split digits into ints)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_cctv_videos() -> List[Path]:
    """Supported video files in CCTVS/, naturally sorted by filename.

    Single source of truth shared by /api/cameras and /api/search so the feeds
    shown in the UI are exactly the videos the recognition search scans.
    """
    if not Settings.VIDEO_DIR.exists():
        return []
    files = [
        p for p in Settings.VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]
    return sorted(files, key=lambda p: _natural_key(p.name))

# ============================================================================
# IN-MEMORY CACHE FOR SEARCH RESULTS
# ============================================================================
# On Render (ephemeral filesystem), keep results in memory instead of disk
# This ensures results persist for the request lifecycle and API calls
search_results_cache: Dict[str, Dict[str, Any]] = {}

# Initialize FastAPI app
app = FastAPI(
    title="DRISTI - Lost Person Detection",
    description="Find lost persons in CCTV footage using face recognition",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = SearchService()

# Live-camera placeholder routes (Phase 8 — not implemented yet)
try:
    from src.backend.routes_camera import router as camera_router, search_router as live_search_router
    app.include_router(camera_router)
    app.include_router(live_search_router)
    logger.info("Camera placeholder routes registered")
except Exception as e:
    logger.warning(f"Could not register camera routes: {e}")


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "DRISTI Lost Person Detection v2.0"
    }


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.post("/api/search")
async def search_lost_person(
    file: UploadFile = File(...),
    use_cctv: bool = Query(False, description="Use connected CCTV cameras")
):
    """
    Upload a photo of a lost person and search in CCTV videos.
    Runs synchronously using asyncio.to_thread() to avoid Render timeout issues.
    
    Returns:
        - Results with matched faces
        - Confidence percentages
        - Timestamps and camera info
    """
    try:
        # Validate file
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file provided"}
            )
        
        # Save uploaded file
        upload_path = Settings.UPLOAD_DIR / f"lost_person_{datetime.now().timestamp()}.jpg"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create unique search ID
        search_id = f"search_{datetime.now().timestamp()}"
        
        # Get all available video files (same discovery as /api/cameras)
        videos = list_cctv_videos()
        
        if not videos:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No CCTV video files found in the system",
                    "search_id": search_id
                }
            )
        
        # Run search synchronously using asyncio.to_thread() to prevent timeout
        # This executes the CPU-heavy task in a thread pool without blocking the event loop
        logger.info(f"Starting face search for {len(videos)} video file(s)...")
        results = await asyncio.to_thread(
            search_service.search_in_videos,
            str(upload_path),
            videos,
            search_id
        )
        
        # Store results in memory cache (Render-compatible, no disk dependency)
        search_results_cache[search_id] = results
        logger.info(f"Results cached in memory for search_id: {search_id}")

        # Return results directly (status 200)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "search_id": search_id,
                "results": results,
                "message": "Search completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Search error: {str(e)}"}
        )


@app.get("/api/search-results/{search_id}")
async def get_search_results(search_id: str):
    """
    Retrieve search results from in-memory cache.
    Render-compatible: uses memory cache instead of disk reads.
    """
    try:
        # Check memory cache first (primary source)
        if search_id in search_results_cache:
            results = search_results_cache[search_id]
            logger.info(f"Found results in memory cache for search_id: {search_id}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "search_id": search_id,
                    "results": results
                }
            )
        
        # Fallback: Try to read from disk (for recovered sessions, if disk available)
        results_file = Settings.RESULTS_DIR / f"{search_id}.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
                # Cache it in memory for future requests
                search_results_cache[search_id] = results
                logger.info(f"Loaded results from disk for search_id: {search_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "search_id": search_id,
                        "results": results
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read results file: {e}")
        
        # Results not found in memory or disk
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Search results not found. Search may still be processing.",
                "search_id": search_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error retrieving results: {str(e)}",
                "search_id": search_id
            }
        )


@app.post("/api/clear-results")
async def clear_results():
    """
    Clear generated snapshot image files from the results directory
    and clear the in-memory search results cache. Called when user
    starts a new search to free up ephemeral storage.
    """
    try:
        # Clear in-memory cache
        search_results_cache.clear()

        # Remove image files from results directory (jpg/png/jpeg/png)
        deleted_files = []
        if Settings.RESULTS_DIR.exists():
            for f in Settings.RESULTS_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    try:
                        f.unlink()
                        deleted_files.append(f.name)
                    except Exception as e:
                        logger.warning(f"Could not delete result file {f}: {e}")

        logger.info(f"Cleared {len(deleted_files)} result file(s) from disk and cleared cache")
        return JSONResponse(status_code=200, content={"success": True, "deleted": deleted_files})

    except Exception as e:
        logger.error(f"Error clearing results: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/api/snapshot/{filename}")
async def get_snapshot(filename: str):
    """
    Retrieve snapshot image from search results.
    On Render: snapshots may not exist due to ephemeral filesystem.
    Returns placeholder if snapshot unavailable.
    """
    try:
        snapshot_path = Settings.RESULTS_DIR / filename
        
        # Try to return snapshot if it exists. no-store so the browser never
        # shows a stale snapshot from a previous (live) search.
        if snapshot_path.exists():
            logger.info(f"Serving snapshot: {filename}")
            return FileResponse(snapshot_path, media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
        
        # Snapshot not found (expected on Render due to ephemeral filesystem)
        logger.warning(f"Snapshot not found: {filename}. Using placeholder.")
        
        # Return JSON response instead of 404
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Snapshot not available (ephemeral storage on serverless platform)",
                "filename": filename
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving snapshot {filename}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error retrieving snapshot: {str(e)}",
                "filename": filename
            }
        )


# ============================================================================
# CAMERA MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/cameras")
async def get_cameras():
    """List local CCTV feeds discovered in CCTVS/ (no remote/Cloudinary URLs).

    `id` is the file stem (matches `match.camera` from search results), so the
    frontend can map result badges onto the correct dynamically-rendered card.
    """
    cameras = []
    for idx, video_file in enumerate(list_cctv_videos(), start=1):
        cameras.append({
            "id": video_file.stem,                              # e.g. "cctv4"
            "name": f"CCTV {idx}",                              # friendly label
            "filename": video_file.name,
            "video_url": f"/api/cctv-video/{video_file.name}",  # served locally
            "source": "local",
        })

    return {
        "success": True,
        "cameras": cameras,
        "total": len(cameras),
    }


@app.get("/api/cctv-video/{filename}")
async def get_cctv_video(filename: str):
    """Safely serve a single video file from CCTVS/ only.

    Guards against path traversal and unsupported extensions; supports HTTP
    range requests (via Starlette FileResponse) so the browser can stream/seek.
    """
    # 1. Must be a bare filename — reject any path component / traversal attempt.
    if (not filename or filename != os.path.basename(filename)
            or "/" in filename or "\\" in filename or filename in (".", "..")):
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid filename"})

    # 2. Extension allowlist.
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTS:
        return JSONResponse(status_code=400, content={"success": False, "message": "Unsupported file type"})

    # 3. Resolve and confirm the file lives directly inside CCTVS/.
    base = Settings.VIDEO_DIR.resolve()
    target = (base / filename).resolve()
    if target.parent != base:
        return JSONResponse(status_code=403, content={"success": False, "message": "Forbidden"})
    if not target.is_file():
        return JSONResponse(status_code=404, content={"success": False, "message": "Video not found"})

    return FileResponse(target, media_type=_VIDEO_MEDIA_TYPES.get(ext, "application/octet-stream"))


@app.get("/api/cctv/config")
async def get_cctv_config():
    """Get saved CCTV camera configuration"""
    try:
        config_file = Path("cctv_config.json")
        
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
            
            return {
                "success": True,
                "cameras": config
            }
        
        return {"success": True, "cameras": []}
        
    except Exception as e:
        logger.error(f"Error reading CCTV config: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@app.post("/api/cctv/config")
async def save_cctv_config(cameras: Dict[str, Any]):
    """Save CCTV camera configuration"""
    try:
        config_file = Path("cctv_config.json")
        
        with open(config_file, "w") as f:
            json.dump(cameras, f, indent=2)
        
        logger.info(f"CCTV config saved: {len(cameras)} camera(s)")
        
        return {
            "success": True,
            "message": f"Configuration saved for {len(cameras)} camera(s)"
        }
    except Exception as e:
        logger.error(f"Error saving CCTV config: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


# ============================================================================
# STATIC FILES
# ============================================================================

# v2 (M1, task A2): register every /api/v2 router BEFORE the static mount.
# Anything included after app.mount("/") is unreachable (docs/TRD.md §2 quirk 1).
try:
    from src.backend.v2 import register_v2
    register_v2(app)
    logger.info("v2 routes registered")
except Exception as e:
    logger.warning(f"Could not register v2 routes: {e}", exc_info=True)

# Mount frontend static files
try:
    app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
    logger.info("Frontend mounted successfully")
except Exception as e:
    logger.warning(f"Could not mount frontend: {e}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup - Create required directories"""
    # Create necessary directories
    Settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    Settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("DRISTI - Lost Person Detection System v2.0")
    logger.info("=" * 70)
    logger.info(f"Upload directory: {Settings.UPLOAD_DIR}")
    logger.info(f"Results directory: {Settings.RESULTS_DIR}")
    logger.info(f"Videos directory: {Settings.VIDEO_DIR}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("DRISTI system shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Settings.PORT)
