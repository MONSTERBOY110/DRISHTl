"""
Camera worker thread.

ONE cv2.VideoCapture per camera. The thread continuously reads frames and:
  - keeps the latest frame as a small, downscaled JPEG for live preview,
    re-encoded only at LIVE_PREVIEW_FPS (not on every decoded frame — this was
    the main CPU/lag culprit),
  - at LIVE_PROCESS_FPS (default 1) runs InsightFace + quality gate on a frame
    and pushes valid embeddings into the time-based rolling buffer.

Preview and search both read from this single worker (the stream is never
opened a second time). Survives transient disconnects by reopening.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2

from src.backend.vision import config, matcher, quality
from src.backend.camera.rolling_buffer import RollingBuffer

logger = logging.getLogger(__name__)

_MAX_REOPEN_FAILURES = 5


def _downscale(frame, width: int):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


class CameraWorker:
    def __init__(self, camera_id: str, name: str, stream_url: str,
                 process_fps: float = None, buffer_seconds: int = None):
        self.camera_id = camera_id
        self.name = name
        self.stream_url = stream_url
        self.process_fps = max(0.1, float(process_fps or config.LIVE_PROCESS_FPS))
        self.preview_fps = max(1.0, float(config.LIVE_PREVIEW_FPS))
        self.buffer = RollingBuffer(camera_id, buffer_seconds or config.LIVE_BUFFER_SECONDS)

        self.status = "connecting"      # connecting | connected | disconnected | error
        self.error: Optional[str] = None
        self.processed_frames = 0       # frames sent to recognition
        self.frames_read = 0            # raw frames decoded
        self.last_frame_time: Optional[float] = None
        self.connected_at: Optional[float] = None

        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{camera_id}", daemon=True)

    # ------------------------------------------------------------------ #
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def wait_until_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.status in ("connected", "error", "disconnected"):
                return self.status == "connected"
            time.sleep(0.1)
        return self.status == "connected"

    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def status_dict(self) -> Dict[str, Any]:
        now = time.time()
        age = round(now - self.last_frame_time, 2) if self.last_frame_time else None
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "stream_url": self.stream_url,
            "status": self.status,
            "connected": self.status == "connected",
            "last_error": self.error,
            "processed_frames": self.processed_frames,
            "frames_read": self.frames_read,
            "buffer_entries": len(self.buffer),
            "stored_embeddings": self.buffer.face_count(),
            "buffered_frames": len(self.buffer),       # back-compat alias
            "buffered_faces": self.buffer.face_count(),# back-compat alias
            "latest_frame_age_seconds": age,
            "preview_fps": self.preview_fps,
            "process_fps": self.process_fps,
            "buffer_seconds": self.buffer.buffer_seconds,
            "uptime_seconds": round(now - self.connected_at, 1) if self.connected_at else 0,
        }

    # ------------------------------------------------------------------ #
    def _open(self):
        cap = cv2.VideoCapture(self.stream_url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _run(self):
        cap = self._open()
        if not cap.isOpened():
            self.status = "error"
            self.error = "Could not open stream URL"
            logger.error(f"[cam {self.camera_id}] cannot open {self.stream_url}")
            return

        reopen_failures = 0
        next_process = 0.0
        next_preview = 0.0
        process_interval = 1.0 / self.process_fps
        preview_interval = 1.0 / self.preview_fps

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                reopen_failures += 1
                logger.warning(f"[cam {self.camera_id}] read failed ({reopen_failures}/{_MAX_REOPEN_FAILURES})")
                if reopen_failures >= _MAX_REOPEN_FAILURES:
                    self.status = "disconnected"
                    self.error = "Stream ended or unreachable"
                    break
                cap.release()
                time.sleep(0.5)
                cap = self._open()
                continue

            reopen_failures = 0
            self.frames_read += 1
            now = time.time()
            self.last_frame_time = now
            if self.status == "connecting":
                self.status = "connected"
                self.connected_at = now
                logger.info(f"[cam {self.camera_id}] connected: {self.stream_url}")

            # Preview: downscale + encode only at preview FPS (NOT every frame).
            if now >= next_preview:
                next_preview = now + preview_interval
                small = _downscale(frame, config.LIVE_PREVIEW_WIDTH)
                ok, buf = cv2.imencode(".jpg", small,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), config.LIVE_PREVIEW_JPEG_QUALITY])
                if ok:
                    with self._jpeg_lock:
                        self._latest_jpeg = buf.tobytes()

            # Recognition: only at the (much lower) process FPS.
            if now >= next_process:
                next_process = now + process_interval
                try:
                    self._index_frame(frame, now)
                except Exception as e:  # never let processing kill the capture loop
                    logger.warning(f"[cam {self.camera_id}] index error: {e}")

        cap.release()
        if self.status not in ("disconnected", "error"):
            self.status = "disconnected"
        logger.info(f"[cam {self.camera_id}] worker stopped (status={self.status})")

    def _index_frame(self, frame, timestamp: float):
        """Detect + quality-gate + embed faces and push them to the buffer."""
        from src.backend.vision import detector  # lazy: avoids load before first use

        faces = detector.extract_faces(frame)
        valid: List[Dict[str, Any]] = []
        for f in faces:
            ok, _ = quality.is_valid_face(f, frame)
            if not ok:
                continue
            emb = matcher.normalize(f.get("embedding"))
            if emb is None:
                continue
            valid.append({"bbox": [int(v) for v in f["bbox"]],
                          "det_score": float(f.get("det_score", 0.0)),
                          "embedding": emb})

        # Store the full-res JPEG (bbox coords are full-res; used only if this
        # frame becomes a match snapshot). Encoded at 1 FPS, so cost is low.
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_jpeg = buf.tobytes() if ok else b""
        self.processed_frames += 1
        self.buffer.add(timestamp, self.processed_frames, frame_jpeg, valid)
