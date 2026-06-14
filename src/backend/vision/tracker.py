"""
Simple IoU-based face tracker + per-track aggregation.

Within a single video, each detected face box is matched to the most-overlapping
box from the previous processed frame (IoU >= IOU_THRESHOLD); otherwise it starts
a new track. Each track accumulates the per-frame similarity to the target plus
enough metadata to build a track-level match record (best frame/crop, timestamp
span). Final match decisions are made per track, not per frame.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """Accumulates detections believed to belong to one person."""

    __slots__ = (
        "track_id", "last_bbox", "similarities", "timestamps",
        "best_similarity", "best_frame_img", "best_crop_img",
        "best_frame_number", "best_bbox", "ts_start", "ts_end",
    )

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.last_bbox = bbox
        self.similarities: List[float] = []
        self.timestamps: List[float] = []
        self.best_similarity: float = -1.0
        self.best_frame_img: Optional[np.ndarray] = None
        self.best_crop_img: Optional[np.ndarray] = None
        self.best_frame_number: int = 0
        self.best_bbox: Tuple[int, int, int, int] = bbox
        self.ts_start: float = 0.0
        self.ts_end: float = 0.0

    def update(self, bbox, similarity, timestamp, frame_img, crop_img, frame_number):
        self.last_bbox = bbox
        self.similarities.append(similarity)
        self.timestamps.append(timestamp)
        if not self.timestamps[:-1]:
            self.ts_start = timestamp
        self.ts_end = timestamp
        if similarity > self.best_similarity:
            self.best_similarity = similarity
            self.best_frame_img = frame_img
            self.best_crop_img = crop_img
            self.best_frame_number = frame_number
            self.best_bbox = bbox


class IoUTracker:
    """Greedy per-frame IoU association. Reset per video via a fresh instance."""

    def __init__(self):
        self._tracks: List[Track] = []
        self._next_id = 0
        self._prev_frame_tracks: List[Track] = []

    def assign(self, bbox: Tuple[int, int, int, int]) -> Track:
        """Return the track this bbox belongs to (existing or newly created)."""
        best_track, best_iou = None, 0.0
        for t in self._prev_frame_tracks:
            score = iou(bbox, t.last_bbox)
            if score >= config.IOU_THRESHOLD and score > best_iou:
                best_track, best_iou = t, score
        if best_track is not None:
            return best_track
        track = Track(self._next_id, bbox)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def begin_frame(self):
        """Snapshot current tracks as the association set for the next frame."""
        self._prev_frame_tracks = list(self._tracks)

    @property
    def tracks(self) -> List[Track]:
        return self._tracks
