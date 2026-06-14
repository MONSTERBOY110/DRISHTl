"""
Search Service — public entry point for face search in video files.

This is now a thin router. The heavy ML logic lives in `src.backend.vision`:
  - RECOGNIZER_BACKEND="insightface" (default) -> vision.pipeline.search  (open-set, safe)
  - RECOGNIZER_BACKEND="legacy"               -> vision.legacy_engine     (Haar + ResNet18)

RTSP capture/test helpers are retained here unchanged.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.backend.vision import config

logger = logging.getLogger(__name__)


class SearchService:
    """Routes a search to the configured recognition backend."""

    def __init__(self):
        logger.info(f"SearchService initialized (RECOGNIZER_BACKEND={config.RECOGNIZER_BACKEND})")
        self._legacy_engine = None  # lazy — only built if legacy backend is used

    # ------------------------------------------------------------------ #
    # Public search entry point (called by app.py)
    # ------------------------------------------------------------------ #
    def search_in_videos(self, lost_person_path: str, video_paths: List[Path],
                         search_id: str) -> Dict[str, Any]:
        """Search for the lost person across the given videos."""
        logger.info(f"Starting search {search_id} over {len(video_paths)} video(s)")

        if config.RECOGNIZER_BACKEND == "legacy":
            return self._legacy().search_in_videos(lost_person_path, video_paths, search_id)

        # Default: modern open-set InsightFace pipeline. Fall back to legacy
        # only if the engine cannot be loaded at all (e.g. missing model).
        try:
            from src.backend.vision import pipeline
            return pipeline.search(lost_person_path, video_paths, search_id)
        except Exception as e:
            logger.error(f"InsightFace pipeline failed ({e}); falling back to legacy.",
                         exc_info=True)
            return self._legacy().search_in_videos(lost_person_path, video_paths, search_id)

    def _legacy(self):
        if self._legacy_engine is None:
            from src.backend.vision.legacy_engine import LegacyEngine
            self._legacy_engine = LegacyEngine()
        return self._legacy_engine

    # ------------------------------------------------------------------ #
    # RTSP helpers (unchanged behaviour)
    # ------------------------------------------------------------------ #
    def test_rtsp_connection(self, rtsp_url: str) -> Dict[str, Any]:
        """Test if an RTSP stream is accessible."""
        try:
            logger.info(f"Testing RTSP connection: {rtsp_url[:50]}...")
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                return {"connected": False, "error": "Cannot open RTSP stream"}
            ret, _ = cap.read()
            cap.release()
            if ret:
                return {"connected": True, "message": "Stream is accessible"}
            return {"connected": False, "error": "Cannot read frame from stream"}
        except Exception as e:
            logger.error(f"RTSP test error: {e}")
            return {"connected": False, "error": str(e)}

    def capture_rtsp_footage(self, rtsp_url: str, duration_seconds: int = 15,
                             camera_name: str = "CCTV", output_dir: str = "CCTVS") -> Optional[str]:
        """Capture up to 15 seconds of footage from a live RTSP stream."""
        try:
            duration_seconds = min(duration_seconds, 15)
            logger.info(f"Capturing {duration_seconds}s from {camera_name}...")
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logger.error(f"Failed to connect to RTSP stream: {camera_name}")
                return None

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width == 0 or height == 0:
                logger.error(f"Invalid stream resolution: {width}x{height}")
                cap.release()
                return None

            output_path_obj = Path(output_dir)
            output_path_obj.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in camera_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
            output_file = output_path_obj / f"rtsp_capture_{safe_name}_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error(f"Failed to open video writer for {output_file}")
                cap.release()
                return None

            frame_count = 0
            max_frames = fps * duration_seconds
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Lost connection to {camera_name} after {frame_count} frames")
                    break
                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()
            if frame_count == 0:
                logger.error(f"No frames captured from {camera_name}")
                output_file.unlink(missing_ok=True)
                return None
            logger.info(f"Capture complete: {output_file} ({frame_count} frames)")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error capturing from RTSP {camera_name}: {e}", exc_info=True)
            return None
