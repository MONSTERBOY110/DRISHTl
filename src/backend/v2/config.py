"""
v2 settings: the single tuning surface for everything under /api/v2.

Why one file with headed blocks: four people append settings in parallel
(TRD §8), so each owner keeps their own clearly headed block and never edits
another's. Helpers mirror vision/config.py (_f/_i/_s) so env overrides behave
identically, just with the DRISHTI_V2_ prefix (e.g. DRISHTI_V2_FRAME_SKIP=2).
Face thresholds are NOT duplicated here: reuse vision.config (MATCH_THRESHOLD,
TARGET_MIN_*, ...) so v2 and v0.5 can never disagree about what a match is.
"""

import os
from pathlib import Path


def _f(name: str, default: float) -> float:
    """Read a float from env (prefixed DRISHTI_V2_) or use default."""
    raw = os.getenv(f"DRISHTI_V2_{name}")
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    raw = os.getenv(f"DRISHTI_V2_{name}")
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _s(name: str, default: str) -> str:
    return os.getenv(f"DRISHTI_V2_{name}", default)


# ============================================================================
# M1 — core: paths, indexing, trail
# ============================================================================
VERSION = "2.0"
BASE_DIR = Path(__file__).resolve().parents[3]              # repo root
DATA_DIR = Path(_s("DATA_DIR", str(BASE_DIR / "data" / "v2")))
DB_PATH = DATA_DIR / "drishti_v2.db"
CROPS_DIR = DATA_DIR / "crops"
FRAMES_DIR = DATA_DIR / "frames"
CASES_DIR = DATA_DIR / "cases"
UPLOADS_DIR = DATA_DIR / "uploads"
VIDEO_DIR = BASE_DIR / "CCTVS"                              # same folder as the baseline
CAMERAS_JSON = VIDEO_DIR / "cameras.json"                   # template: docs/cameras.example.json

FRAME_SKIP = _i("FRAME_SKIP", 3)              # index every Nth frame
MIN_TRACK_FRAMES = _i("MIN_TRACK_FRAMES", 3)  # drop shorter person tracks
TOPK_CROPS = _i("TOPK_CROPS", 5)              # per-track person crops kept for the appearance embedding
TOPK_FACES = _i("TOPK_FACES", 5)              # per-track face embeddings kept
TRAIL_MERGE_S = _f("TRAIL_MERGE_S", 30.0)     # merge same-camera trail points closer than this
LEAD_DEDUP_S = _f("LEAD_DEDUP_S", 10.0)       # one lead per camera per window (ByteTrack fragments)
WARM_FACE_MODEL = _i("WARM_FACE_MODEL", 1)    # load InsightFace at startup in a thread (0 on CPU-only boxes)

# ============================================================================
# M2 — ML / vision
# Thresholds below remain the TRD defaults until B5 is measured on demo footage.
# ============================================================================
PERSON_MODEL = _s("PERSON_MODEL", "models/yolo11n.pt")     # keep weights out of the repo root
PERSON_CONF = _f("PERSON_CONF", 0.40)
APPEARANCE_MODEL = _s("APPEARANCE_MODEL", "ViT-B-32")
APPEARANCE_PRETRAINED = _s("APPEARANCE_PRETRAINED", "laion2b_s34b_b79k")
ATTR_BONUS_Z = _f("ATTR_BONUS_Z", 0.75)       # per matching colour slot (upper / lower)
LEAD_MIN_Z = _f("LEAD_MIN_Z", 1.0)
TRAIL_MIN_Z = _f("TRAIL_MIN_Z", 1.5)
LEADS_MAX = _i("LEADS_MAX", 6)
LIVE_DESC_WINDOW_S = _f("LIVE_DESC_WINDOW_S", 12.0)
LIVE_DESC_MAX_FRAMES = _i("LIVE_DESC_MAX_FRAMES", 8)
# Day 2 intake (B7): optional OpenAI-compatible LLM; rules fallback when unset.
LLM_API_KEY = _s("LLM_API_KEY", "")
LLM_BASE_URL = _s("LLM_BASE_URL", "")
LLM_MODEL = _s("LLM_MODEL", "")

# Gemini / Voice Intake (B7)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("DRISHTI_V2_GEMINI_API_KEY", ""))
GEMINI_TRANSCRIBE_MODEL = _s("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
GEMINI_TEXT_MODEL = _s("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
INTAKE_MAX_AUDIO_BYTES = _i("INTAKE_MAX_AUDIO_BYTES", 10 * 1024 * 1024)
INTAKE_TIMEOUT_S = _f("INTAKE_TIMEOUT_S", 15.0)

# ============================================================================
# M4 — watchlist + alerts (defaults from TRD §8; M4 owns and may extend)
# ============================================================================
WATCH_INTERVAL_S = _f("WATCH_INTERVAL_S", 1.0)
WATCH_WINDOW_S = _i("WATCH_WINDOW_S", 10)
WATCH_MIN_HITS = _i("WATCH_MIN_HITS", 2)
ALERT_COOLDOWN_S = _i("ALERT_COOLDOWN_S", 60)
TELEGRAM_BOT_TOKEN = _s("TELEGRAM_BOT_TOKEN", "")          # secrets live in .env only
TELEGRAM_CHAT_ID = _s("TELEGRAM_CHAT_ID", "")
