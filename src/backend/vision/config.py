"""
Vision pipeline configuration — the single tuning surface for face search.

Edit these constants to trade precision vs recall. The defaults favour
PRECISION: it is better to return "No Reliable Match" than to show the wrong
person. A commented high-precision-demo block is provided below.

All values can also be overridden via environment variables of the same name
(e.g. DRISHTI_MATCH_THRESHOLD=0.52) so thresholds can be tuned without editing
code during a live demo.
"""

import os


def _f(name: str, default: float) -> float:
    """Read a float threshold from env (prefixed DRISHTI_) or use default."""
    raw = os.getenv(f"DRISHTI_{name}")
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    raw = os.getenv(f"DRISHTI_{name}")
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _s(name: str, default: str) -> str:
    return os.getenv(f"DRISHTI_{name}", default)


# ============================================================================
# BACKEND SELECTORS  (Phase 7 — future-ready switching)
# ============================================================================
# RECOGNIZER_BACKEND: "insightface" (default) | "legacy" (Haar + ResNet18) | "adaface" (future)
# DETECTOR_BACKEND:   "insightface" (default) | "scrfd" (future)
RECOGNIZER_BACKEND = _s("RECOGNIZER_BACKEND", "insightface").lower()
DETECTOR_BACKEND = _s("DETECTOR_BACKEND", "insightface").lower()

# ============================================================================
# INSIGHTFACE MODEL
# ============================================================================
# buffalo_l = SCRFD detector + ResNet100 ArcFace recognizer (best accuracy).
# Downloads (~330MB) to ~/.insightface/models on first use. Use "buffalo_s"
# for a lighter/faster pack on constrained hosts.
MODEL_PACK = _s("MODEL_PACK", "buffalo_l")
# Detector input size. Smaller = much faster on CPU but misses small/distant
# faces. 640 = accurate (default); 320 = ~4x faster for CPU demos.
_DET = _i("DET_SIZE", 640)
DET_SIZE = (_DET, _DET)

# ============================================================================
# FACE QUALITY GATE  (Phase 2) — applied to faces detected in VIDEO frames.
# Blocks background/blurry crops from being matched, WITHOUT discarding genuine
# but small/soft CCTV faces. Calibration on the bundled footage showed the
# original strict gate (size 55 / blur 80) silently dropped real appearances
# (e.g. De Paul's clear 0.63 frames), so it is loosened here. Background still
# never matches because the detector only emits actual faces and the open-set
# threshold below rejects weak similarities.
# ============================================================================
MIN_DETECTION_SCORE = _f("MIN_DETECTION_SCORE", 0.60)
# size 35 was still rejecting genuine appearances: per-camera diagnostics on the
# cctv5-8 footage showed Mam/Sir faces that scored 0.55-0.63 (no-gate) being
# dropped as "small_faces" (e.g. Sir in cctv5/cctv6 best ~0.61/0.63 ungated, but
# 0.11/0.20 after the gate). Lowered to 24 so distant CCTV faces are scored; the
# 0.55 decision threshold + multi-frame rule still guard against false positives.
MIN_FACE_SIZE = _i("MIN_FACE_SIZE", 24)          # min bbox width AND height (px)
MIN_BLUR_SCORE = _f("MIN_BLUR_SCORE", 25.0)      # Laplacian variance
MIN_BRIGHTNESS = _f("MIN_BRIGHTNESS", 40.0)      # grayscale mean
MAX_BRIGHTNESS = _f("MAX_BRIGHTNESS", 220.0)

# Target (uploaded reference photo) gate — deliberately LENIENT. The user chose
# this photo, so we accept it as long as a real face is confidently detected;
# we only reject genuinely unusable uploads (no face / destroyed image). The
# strict gate above still guards every video frame.
TARGET_MIN_DETECTION_SCORE = _f("TARGET_MIN_DETECTION_SCORE", 0.60)
TARGET_MIN_FACE_SIZE = _i("TARGET_MIN_FACE_SIZE", 40)
TARGET_MIN_BLUR_SCORE = _f("TARGET_MIN_BLUR_SCORE", 10.0)

# ============================================================================
# OPEN-SET MATCH DECISION  (Phase 3 / 4)
# ----------------------------------------------------------------------------
# CALIBRATED on the bundled footage with REAL uploaded target photos (Messi &
# De Paul, both genuinely present). Measured genuine vs. imposter separation
# (cosine on normalised ArcFace embeddings, low-res CCTV faces):
#   De Paul (present)  -> best 0.63 (brief appearance, ~2 sampled frames)
#   Messi   (present)  -> best 0.57, several frames in 0.45-0.55
#   other people       -> never exceed ~0.49
# Genuine band (0.55-0.63) sits just above the imposter band (<=0.49), so:
#   MATCH 0.55  -> only genuine faces ever reach it (imposters cap at 0.49)
#   POSSIBLE 0.45 + 3-frame floor -> catches Messi without admitting imposters
#   MIN_MATCH_FRAMES 2 -> needed so De Paul's brief (2-frame) appearance counts
# This finds BOTH present people while still rejecting non-face / other people.
#
# Threshold guide:
#   0.55 = balanced default (finds the present demo subjects; imposters <=0.49)
#   0.60 = higher precision, but rejected real present people on this footage
#   0.50 = more recall for very low-quality faces, higher false-positive risk
# If genuine matches get missed, lower DRISHTI_MATCH_THRESHOLD and/or
# DRISHTI_FRAME_SKIP (denser sampling catches brief appearances).
# ============================================================================
MATCH_THRESHOLD = _f("MATCH_THRESHOLD", 0.55)            # strong-match cosine floor
POSSIBLE_MATCH_THRESHOLD = _f("POSSIBLE_MATCH_THRESHOLD", 0.45)
MIN_MATCH_FRAMES = _i("MIN_MATCH_FRAMES", 2)             # frames >= MATCH_THRESHOLD for strong
MIN_AVG_SIMILARITY = _f("MIN_AVG_SIMILARITY", 0.48)      # avg of top strong scores

# ============================================================================
# TRACKING & PROCESSING
# ============================================================================
IOU_THRESHOLD = _f("IOU_THRESHOLD", 0.30)   # bbox overlap to keep same track_id
# Lowered 5 -> 3: at skip 5 a moving person's bbox jumps too far between sampled
# frames, so IoU association breaks and one person fragments into many 1-2 frame
# tracks (observed: Sir in cctv7 split across ~26 tracks, none reaching the
# 2-frame floor). Denser sampling keeps tracks intact and captures brief
# appearances. Costs more CPU time per search.
FRAME_SKIP = _i("FRAME_SKIP", 3)            # process every Nth frame
MAX_RESULTS = _i("MAX_RESULTS", 12)         # cap (used only in global RESULT_MODE)

# ============================================================================
# RESULT AGGREGATION
# ----------------------------------------------------------------------------
# "per_camera" (default): return the best track(s) from EACH camera that has a
#   qualifying match, so a person present in N feeds yields N result cards. Fixes
#   the old behaviour where a global top-N could collapse multi-camera presence.
# "global": legacy — pool all tracks, sort, cap at MAX_RESULTS.
# ============================================================================
RESULT_MODE = _s("RESULT_MODE", "per_camera")
MAX_RESULTS_PER_CAMERA = _i("MAX_RESULTS_PER_CAMERA", 2)
# A camera with no qualifying track but a best similarity within this margin of
# POSSIBLE_MATCH_THRESHOLD is reported under "near_misses" (debug aid, not a match).
NEAR_MISS_MARGIN = _f("NEAR_MISS_MARGIN", 0.05)

# ============================================================================
# LIVE CAMERA (phone / IP-webcam, e.g. http://PHONE_IP:8080/video)
# ----------------------------------------------------------------------------
# The live worker processes frames continuously in the background at a low FPS
# and keeps a rolling buffer of face embeddings, so a live search just compares
# vectors (fast) instead of re-running detection over raw footage. Live search
# reuses MATCH_THRESHOLD / POSSIBLE_MATCH_THRESHOLD / MIN_MATCH_FRAMES above.
# ============================================================================
LIVE_PROCESS_FPS = _f("LIVE_PROCESS_FPS", 1.0)        # frames/sec sent to InsightFace
LIVE_PREVIEW_FPS = _f("LIVE_PREVIEW_FPS", 8.0)        # frames/sec encoded for MJPEG preview
LIVE_PREVIEW_WIDTH = _i("LIVE_PREVIEW_WIDTH", 640)    # preview is downscaled to this width
LIVE_PREVIEW_JPEG_QUALITY = _i("LIVE_PREVIEW_JPEG_QUALITY", 70)
LIVE_BUFFER_SECONDS = _i("LIVE_BUFFER_SECONDS", 30)   # max rolling buffer time window
LIVE_SEARCH_DURATION_SECONDS = _i("LIVE_SEARCH_DURATION_SECONDS", 10)  # default recent window searched
LIVE_CONNECT_TIMEOUT = _f("LIVE_CONNECT_TIMEOUT", 10.0)  # secs to confirm first frame on connect

# ============================================================================
# EVEN-HIGHER-PRECISION PRESET  (uncomment to enable; for noisy/crowded feeds)
# ============================================================================
# MIN_DETECTION_SCORE = 0.80
# MATCH_THRESHOLD = 0.65
# MIN_MATCH_FRAMES = 4
# MIN_AVG_SIMILARITY = 0.60
