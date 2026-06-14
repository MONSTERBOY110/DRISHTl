"""
Camera manager — owns live CameraWorker threads.

`connect()` starts a worker and waits briefly to confirm the stream actually
opens (so the frontend gets an immediate success/failure), then leaves the
worker running in the background to fill the rolling buffer.
"""

import logging
from typing import Any, Dict, List, Optional

from src.backend.vision import config
from src.backend.camera.camera_worker import CameraWorker

logger = logging.getLogger(__name__)


class CameraManager:
    """Registry of live cameras and their capture/processing workers."""

    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}

    def connect(self, camera_id: str, name: str, stream_url: str,
                process_fps: float = None, buffer_seconds: int = None) -> Dict[str, Any]:
        """Start (or restart) a camera worker and confirm the stream opens."""
        # Replace any existing worker for this id.
        self.disconnect(camera_id)

        worker = CameraWorker(
            camera_id=camera_id,
            name=name or camera_id,
            stream_url=stream_url,
            process_fps=process_fps or config.LIVE_PROCESS_FPS,
            buffer_seconds=buffer_seconds or config.LIVE_BUFFER_SECONDS,
        )
        self._workers[camera_id] = worker
        worker.start()

        connected = worker.wait_until_ready(config.LIVE_CONNECT_TIMEOUT)
        if not connected:
            err = worker.error or "Stream did not produce a frame in time"
            worker.stop()
            self._workers.pop(camera_id, None)
            return {"success": False, "error": err}

        return {"success": True, "camera": worker.status_dict()}

    def disconnect(self, camera_id: str) -> bool:
        worker = self._workers.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        return True

    def get(self, camera_id: str) -> Optional[CameraWorker]:
        return self._workers.get(camera_id)

    def status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        worker = self._workers.get(camera_id)
        return worker.status_dict() if worker else None

    def list_cameras(self) -> List[Dict[str, Any]]:
        return [w.status_dict() for w in self._workers.values()]


# Process-wide singleton used by the routes.
camera_manager = CameraManager()
