#!/usr/bin/env python3
"""
CLI preview for PersonTracker: draws bounding boxes and track IDs.

Usage:
    python scripts/demo_persons.py <video_path> [--out <output_path>] [--max-frames <N>]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from src.backend.v2.persons import PersonTracker


def main():
    parser = argparse.ArgumentParser(description="Preview YOLO11n + ByteTrack on video.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--out", "-o", default=None, help="Output annotated video path")
    parser.add_argument("--max-frames", "-n", type=int, default=150, help="Max frames to process (default: 150)")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: input video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.out
    if not out_path:
        out_path = str(video_path.parent / f"{video_path.stem}_annotated.mp4")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = PersonTracker(conf=args.conf)
    frame_idx = 0
    t0 = time.time()
    tracks_seen = set()

    print(f"Tracking persons in: {video_path} ({width}x{height} @ {fps:.1f} fps, {total_frames} total frames)")
    print(f"Writing preview to: {out_path} (max frames: {args.max_frames})")

    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracks = tracker.update(frame)
        for track_id, (x1, y1, x2, y2), conf in tracks:
            tracks_seen.add(track_id)
            color = (0, 255, 0) if track_id >= 0 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"Frame {frame_idx} | Tracks: {len(tracks)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)
        if frame_idx % 25 == 0:
            print(f"Processed {frame_idx}/{args.max_frames} frames ({len(tracks)} active tracks in current frame)...")

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.2f}s ({frame_idx / max(elapsed, 0.001):.1f} fps).")
    print(f"Unique tracks seen: {len(tracks_seen)}")
    print(f"Annotated preview saved: {out_path}")


if __name__ == "__main__":
    main()

