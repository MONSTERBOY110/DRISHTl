"""
Legacy recognizer (Haar Cascade + ImageNet ResNet18 + cosine).

Preserved verbatim as a fallback for RECOGNIZER_BACKEND="legacy". This is the
ORIGINAL pipeline and is known to produce false positives; it is kept only so
the app still functions if InsightFace cannot be loaded. The result shape mirrors
the modern pipeline's backward-compat fields so the frontend keeps working.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config.settings import Settings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class LegacyEngine:
    """Haar + ResNet18 face search (legacy fallback)."""

    def __init__(self):
        logger.info("Initializing LegacyEngine (Haar + ResNet18)...")
        import torch
        self._torch = torch
        self.device = torch.device("cpu")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.similarity_threshold = 0.6
        self.frame_skip = 5
        self.face_model = self._load_face_model()
        logger.info("LegacyEngine initialized.")

    def _load_face_model(self):
        try:
            import torch.nn as nn
            import torchvision.models as models
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
            logger.info("ResNet18 face model loaded")
            return model
        except Exception as e:
            logger.warning(f"Could not load ResNet: {e}. Using histogram fallback.")
            return None

    @staticmethod
    def _histogram_features(face_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        try:
            if self.face_model is None:
                return self._histogram_features(face_image)
            import torchvision.transforms as transforms
            from PIL import Image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                embedding = self.face_model(tensor)
            embedding = embedding.squeeze().cpu().numpy()
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            logger.warning(f"Embedding error: {e}. Using fallback.")
            return self._histogram_features(face_image)

    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                   minNeighbors=5, minSize=(30, 30))
        out = []
        h, w, _ = frame.shape
        for (x, y, fw, fh) in faces:
            x1, y1 = max(0, x - 15), max(0, y - 15)
            x2, y2 = min(w, x + fw + 15), min(h, y + fh + 15)
            roi = frame[y1:y2, x1:x2].copy()
            if roi.size > 0:
                out.append({"bbox": (x1, y1, x2, y2), "face": roi, "confidence": 0.9})
        return out

    def calculate_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        a = e1 / (np.linalg.norm(e1) + 1e-8)
        b = e2 / (np.linalg.norm(e2) + 1e-8)
        return (float(np.dot(a, b)) + 1) / 2

    @staticmethod
    def _format_time(seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def search_in_videos(self, lost_person_path: str, video_paths: List[Path],
                         search_id: str) -> Dict[str, Any]:
        lost_img = cv2.imread(lost_person_path)
        if lost_img is None:
            return {"search_id": search_id, "status": "error", "decision": "no_reliable_match",
                    "error": "Could not read uploaded image", "matches": [],
                    "stats": {"matches_found": 0}}

        faces = self.detect_faces_in_frame(lost_img)
        if not faces:
            return {"search_id": search_id, "status": "error", "decision": "no_reliable_match",
                    "error": "No face detected in the uploaded photo", "matches": [],
                    "stats": {"matches_found": 0}}

        target_emb = self.extract_face_embedding(faces[0]["face"])
        matches: List[Dict[str, Any]] = []

        for video_path in video_paths:
            video_path = Path(video_path)
            if not video_path.exists():
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_idx = 0
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1
                frame_count += 1
                for fd in self.detect_faces_in_frame(frame):
                    emb = self.extract_face_embedding(fd["face"])
                    sim = self.calculate_similarity(target_emb, emb)
                    if sim >= self.similarity_threshold:
                        ts = (frame_idx / fps) if fps > 0 else 0
                        snap = f"{search_id}_{video_path.stem}_{frame_count}.jpg"
                        matches.append({
                            "camera": video_path.stem,
                            "camera_name": video_path.stem.replace("_", " ").title(),
                            "confidence": round(sim * 100, 2),
                            "timestamp": round(ts, 2),
                            "frame_number": frame_count,
                            "snapshot": snap,
                            "time_formatted": self._format_time(ts),
                        })
            cap.release()

        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "search_id": search_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "decision": "match_found" if matches else "no_reliable_match",
            "matches": matches,
            "stats": {"matches_found": len(matches)},
            "rejected": {},
        }
