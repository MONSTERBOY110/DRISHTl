"""
Time-based rolling buffer of PROCESSED face data for one live camera.

Each entry is one processed frame: capture timestamp, a monotonic frame_id, a
small JPEG of the frame (for snapshot output), and the valid faces found in it
(embedding + bbox + det_score). Entries older than `buffer_seconds` are pruned,
so memory stays bounded (≈ buffer_seconds × process_fps entries). We never store
unbounded raw frames.

`recent()` returns only entries within the last `duration_seconds`, so a live
search can be restricted to a fresh time window (no stale matches).
"""

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class RollingBuffer:
    """Thread-safe, time-windowed buffer of processed face entries."""

    def __init__(self, camera_id: str, buffer_seconds: int = 30):
        self.camera_id = camera_id
        self.buffer_seconds = buffer_seconds
        self._entries: deque = deque()
        self._lock = threading.Lock()
        self.total_indexed = 0  # lifetime count of frames pushed (monotonic)

    def add(self, timestamp: float, frame_id: int, frame_jpeg: bytes,
            faces: List[Dict[str, Any]]) -> None:
        """Add one processed frame's faces; prune anything older than the window."""
        with self._lock:
            self._entries.append({
                "timestamp": timestamp,
                "frame_id": frame_id,
                "frame_jpeg": frame_jpeg,
                "faces": faces,
            })
            self.total_indexed += 1
            self._prune_locked(time.time())

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.buffer_seconds
        while self._entries and self._entries[0]["timestamp"] < cutoff:
            self._entries.popleft()

    def snapshot(self) -> List[Dict[str, Any]]:
        """All current (non-expired) entries."""
        with self._lock:
            self._prune_locked(time.time())
            return list(self._entries)

    def recent(self, duration_seconds: float, now: Optional[float] = None
               ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Return (entries_within_window, total_entries, dropped_older_than_window).

        Only entries with timestamp >= now - duration_seconds are returned, so a
        search never reuses faces older than the requested window.
        """
        if now is None:
            now = time.time()
        with self._lock:
            self._prune_locked(now)
            total = len(self._entries)
            cutoff = now - duration_seconds
            within = [e for e in self._entries if e["timestamp"] >= cutoff]
            return within, total, total - len(within)

    def latest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._entries[-1]["timestamp"] if self._entries else None

    def face_count(self) -> int:
        with self._lock:
            return sum(len(e["faces"]) for e in self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
