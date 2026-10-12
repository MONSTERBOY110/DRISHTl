"""
Generate a synthetic CCTV-like clip so the v2 pipeline can be smoke-tested
with no real footage (and with no faces, so it never depends on the models).

Writes CCTVS/synthetic_cam.mp4 (git-ignored): a static grey "pandal" with a
"person" in a yellow top and blue trousers walking left → right, then a
distractor in a red top and black trousers walking right → left. Searching
"yellow top, blue jeans" should surface the first one as a lead.

Usage: python scripts/make_synthetic_clip.py [out.mp4] [--seconds 8]
Delete it from CCTVS/ before indexing the real demo footage.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 360, 25


def _draw_person(frame, x, y, top_bgr, bottom_bgr, w=48, h=150):
    head_r = w // 4
    cv2.circle(frame, (x + w // 2, y + head_r), head_r, (120, 160, 210), -1)        # head
    cv2.rectangle(frame, (x, y + 2 * head_r), (x + w, y + int(h * 0.55)), top_bgr, -1)     # top
    cv2.rectangle(frame, (x, y + int(h * 0.55)), (x + w, y + h), bottom_bgr, -1)           # trousers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(Path("CCTVS") / "synthetic_cam.mp4"))
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(args.seconds * FPS)
    rng = np.random.default_rng(0)
    background = np.full((H, W, 3), 110, dtype=np.uint8)
    cv2.rectangle(background, (0, int(H * 0.75)), (W, H), (90, 90, 90), -1)          # "floor"
    for i in range(0, W, 80):                                                           # "pillars"
        cv2.rectangle(background, (i, 40), (i + 12, int(H * 0.75)), (130, 125, 120), -1)

    for f in range(n):
        frame = background.copy()
        t = f / max(1, n - 1)
        # target: yellow top, blue trousers, walks left -> right across the first 70 % of the clip
        if t < 0.7:
            x = int(-60 + (W + 60) * (t / 0.7))
            _draw_person(frame, x, 110, (0, 215, 245), (200, 90, 20))
        # distractor: red top, black trousers, right -> left during the last 55 %
        if t > 0.45:
            x = int(W - (W + 60) * ((t - 0.45) / 0.55))
            _draw_person(frame, x, 120, (30, 30, 220), (25, 25, 25))
        noise = rng.integers(-4, 5, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    print(f"wrote {out} ({n} frames, {W}x{H} @ {FPS} fps)")


if __name__ == "__main__":
    main()
