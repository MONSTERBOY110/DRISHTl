"""
Appearance embeddings + clothing colours via OpenCLIP.

Uses OpenCLIP ViT-B-32 (laion2b_s34b_b79k) for joint text↔image embeddings (512-D),
L2-normalised float32, lazy singleton, CUDA if available.
Fallback to padded hue histograms if OpenCLIP is loading or unavailable.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

MODEL_ID = "ViT-B-32/laion2b_s34b_b79k"
DIM = 512

_HUE_BINS = 32
COLOR_NAMES = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = None
_clip_initialized = False


def _init_openclip() -> bool:
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device, _clip_initialized
    if _clip_initialized:
        return _clip_model is not None

    _clip_initialized = True
    try:
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading OpenCLIP %s on %s...", MODEL_ID, device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer
        _clip_device = device
        logger.info("OpenCLIP loaded successfully on %s", device)
        return True
    except Exception as e:
        logger.warning("Could not load OpenCLIP model (%s); falling back to histogram embeddings", e)
        return False


def embed_text(text: str) -> np.ndarray:
    """(DIM,) float32 L2-normalised text embedding for description search."""
    if not text or not text.strip():
        return np.zeros(DIM, dtype=np.float32)

    if _init_openclip():
        try:
            import torch
            with torch.no_grad():
                tokens = _clip_tokenizer([text]).to(_clip_device)
                feat = _clip_model.encode_text(tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().squeeze(0).astype(np.float32)
        except Exception as e:
            logger.error("Error in OpenCLIP embed_text: %s", e)

    return np.zeros(DIM, dtype=np.float32)


def embed_images(crops: List[np.ndarray]) -> np.ndarray:
    """(N, DIM) float32 L2-normalised crop embeddings for indexed person tracks."""
    if not crops:
        return np.zeros((0, DIM), dtype=np.float32)

    if _init_openclip():
        try:
            import torch
            from PIL import Image

            pil_images = []
            for crop in crops:
                if crop is None or crop.size == 0:
                    crop = np.zeros((64, 64, 3), dtype=np.uint8)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_images.append(_clip_preprocess(Image.fromarray(rgb)))

            tensor_batch = torch.stack(pil_images).to(_clip_device)
            with torch.no_grad():
                feats = _clip_model.encode_image(tensor_batch)
                feats /= feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error("Error in OpenCLIP embed_images: %s", e)

    # Fallback to 512-D padded hue histograms if model fails
    rows = []
    for crop in crops:
        v = _hue_hist_512(crop)
        n = float(np.linalg.norm(v))
        rows.append(v / n if n > 0 else v)
    return np.stack(rows).astype(np.float32)


def _hue_hist_512(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros(DIM, dtype=np.float32)
    h = crop.shape[0]
    upper = crop[: max(1, h // 2)]
    lower = crop[h // 2:]

    def _hist(reg):
        if reg is None or reg.size == 0:
            return np.zeros(_HUE_BINS, dtype=np.float32)
        hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
        hist, _ = np.histogram(hsv[..., 0], bins=_HUE_BINS, range=(0, 180), weights=hsv[..., 1] / 255.0)
        return hist.astype(np.float32)

    v = np.concatenate([_hist(upper), _hist(lower)])
    padded = np.zeros(DIM, dtype=np.float32)
    padded[: len(v)] = v
    return padded


def _name_color(h: float, s: float, v: float) -> str:
    """Map an OpenCV HSV triple (h 0-180, s/v 0-255) to one of COLOR_NAMES."""
    if v < 50:
        return "black"
    if s < 45:
        return "white" if v > 170 else "gray"
    if h < 8 or h >= 165:
        return "brown" if v < 120 else "red"
    if h < 20:
        return "brown" if v < 130 else "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def _dominant(region_bgr: np.ndarray) -> str:
    if region_bgr is None or region_bgr.size == 0:
        return "gray"
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s_med, v_med = float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))
    chromatic = hsv[(hsv[:, 1] >= 45) & (hsv[:, 2] >= 50)]
    if len(chromatic) < 0.35 * len(hsv):
        return _name_color(0.0, s_med, v_med)
    hist, edges = np.histogram(chromatic[:, 0], bins=_HUE_BINS, range=(0, 180))
    k = int(np.argmax(hist))
    h_mode = float((edges[k] + edges[k + 1]) / 2)
    in_bin = chromatic[(chromatic[:, 0] >= edges[k]) & (chromatic[:, 0] < edges[k + 1])]
    return _name_color(h_mode, float(np.median(in_bin[:, 1])), float(np.median(in_bin[:, 2])))


def color_attributes(person_crop_bgr: np.ndarray) -> Dict[str, str]:
    """{"upper": name, "lower": name} from torso (15-50 %) and legs (55-90 %) bands."""
    if person_crop_bgr is None or person_crop_bgr.size == 0:
        return {"upper": "gray", "lower": "gray"}
    h, w = person_crop_bgr.shape[:2]
    x0, x1 = int(w * 0.2), max(int(w * 0.2) + 1, int(w * 0.8))
    upper = person_crop_bgr[int(h * 0.15): max(int(h * 0.15) + 1, int(h * 0.50)), x0:x1]
    lower = person_crop_bgr[int(h * 0.55): max(int(h * 0.55) + 1, int(h * 0.90)), x0:x1]
    return {"upper": _dominant(upper), "lower": _dominant(lower)}
