"""
Model loading and latency verification script (M2 task B1).

Loads:
- InsightFace buffalo_l
- YOLO11n (models/yolo11n.pt)
- OpenCLIP ViT-B-32 (laion2b_s34b_b79k)

Prints active device and measured load + inference latencies.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def check_models():
    print("=" * 60)
    print("Drishti 2.0 - Model Environment and Latency Check (Task B1)")
    print("=" * 60)

    # 1. InsightFace buffalo_l
    print("\n[1/3] Checking InsightFace buffalo_l...")
    t0 = time.time()
    from src.backend.vision import insightface_engine
    app = insightface_engine.get_face_app()
    load_time_face = (time.time() - t0) * 1000

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    t0 = time.time()
    faces = app.get(dummy_frame)
    inf_time_face = (time.time() - t0) * 1000

    provider = getattr(insightface_engine, "_active_provider", "CPUExecutionProvider")
    print(f"  Provider: {provider}")
    print(f"  Load latency: {load_time_face:.1f} ms")
    print(f"  Dummy inference (640x640): {inf_time_face:.1f} ms")

    # 2. Ultralytics YOLO11n
    print("\n[2/3] Checking YOLO11n person detector...")
    t0 = time.time()
    from ultralytics import YOLO
    import torch
    yolo_path = Path("models/yolo11n.pt")
    if not yolo_path.is_file():
        raise FileNotFoundError(f"Missing YOLO weights at {yolo_path}")
    device = "0" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(str(yolo_path))
    load_time_yolo = (time.time() - t0) * 1000

    t0 = time.time()
    results = yolo_model.track(
        dummy_frame,
        persist=True,
        classes=[0],
        conf=0.40,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
    )
    inf_time_yolo = (time.time() - t0) * 1000
    print(f"  Device: {device}")
    print(f"  Load latency: {load_time_yolo:.1f} ms")
    print(f"  Track inference (640x640): {inf_time_yolo:.1f} ms")

    # 3. OpenCLIP ViT-B-32
    print("\n[3/3] Checking OpenCLIP ViT-B-32...")
    t0 = time.time()
    from src.backend.v2 import appearance
    appearance._init_openclip()
    load_time_clip = (time.time() - t0) * 1000

    t0 = time.time()
    crop_dummy = [np.zeros((128, 64, 3), dtype=np.uint8)]
    img_feat = appearance.embed_images(crop_dummy)
    txt_feat = appearance.embed_text("person in yellow top and blue jeans")
    inf_time_clip = (time.time() - t0) * 1000
    print(f"  Model ID: {appearance.MODEL_ID} (DIM={appearance.DIM})")
    print(f"  Load latency: {load_time_clip:.1f} ms")
    print(f"  Inference latency (crop + text): {inf_time_clip:.1f} ms")
    print(f"  Embeddings L2 norm: img={np.linalg.norm(img_feat[0]):.4f}, txt={np.linalg.norm(txt_feat):.4f}")

    print("\n" + "=" * 60)
    print("ALL THREE MODELS LOADED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    check_models()
