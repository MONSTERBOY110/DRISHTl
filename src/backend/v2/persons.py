"""YOLO person detection with video-local ByteTrack identities.

Why: index-once search needs stable person boxes even when faces are turned
away. Each video gets its own tracker instance so IDs do not leak between cams.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import config

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]
Detection = Tuple[int, Box, float]


class PersonTracker:
    """Lazy YOLO11n + ByteTrack wrapper; create one instance per video."""

    def __init__(self) -> None:
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from ultralytics import YOLO

        weights = Path(config.PERSON_MODEL)
        if not weights.is_absolute():
            weights = config.BASE_DIR / weights
        weights = weights.resolve()
        if weights.parent == config.BASE_DIR:
            raise ValueError(f"Person weights must be stored under models/, not the repo root: {weights}")

        self._device = "0" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(str(weights))
        logger.info("YOLO person tracker ready: weights=%s device=%s", weights, self._device)

    def update(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return tracked class-0 boxes as ``(track_id, xyxy, confidence)``."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return []

        self._load()
        results = self._model.track(
            frame_bgr,
            persist=True,
            classes=[0],
            conf=config.PERSON_CONF,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self._device,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        return [
            (
                int(track_id),
                tuple(int(round(float(v))) for v in coordinates),
                float(confidence),
            )
            for track_id, coordinates, confidence in zip(track_ids, xyxy, confidences)
        ]
