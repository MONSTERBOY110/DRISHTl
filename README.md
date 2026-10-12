# Drishti — find a lost person in the crowd from a photo *or just a description*

At a Durga Puja pandal a child gets separated from her family. Her mother has one old photo and can tell you what the child is wearing. Most CCTV never sees a face, because cameras are high and people face away.

Drishti indexes every camera's footage **once** into person tracks. Each track stores face embeddings when a face was visible, and clothing appearance always. A search is then a few vector products instead of re-scanning video:
- **A photo** gives face-verified sightings.
- **A description** like "yellow top, blue jeans" gives ranked *leads*, even when the face is never visible.
- The sightings are joined into a cross-camera *last seen* trail.

Leads are always labelled **unverified**. There is no fake "confidence %", and "No reliable match" is a valid answer.

## Features

| Feature | Where | Status |
|---|---|---|
| Photo → face search over recorded CCTV (open-set: Strong / Possible / No reliable match) | v0.5 core · `POST /api/search` · classic UI at `/` | built (v0.5) |
| Phone / IP cameras as live feeds | v0.5 core · `/api/camera/*` | built (v0.5) |
| **Index once, search instantly** (YOLO11n + ByteTrack people, InsightFace faces, OpenCLIP appearance) | `src/backend/v2/{indexer,store,search}.py` · `POST /api/v2/search` | built |
| **Search by description, no face needed** (OpenCLIP text↔image + clothing-colour attributes) | `query_parser.py`, `appearance.py` | built, but broken for most queries right now: see Known issues 1 |
| **Cross-camera "last seen" trail** with clock time and venue map | `search.build_trail` · `/v2/` | built |
| **Missing-person cases** (register once with photo + description, search per case) | `/api/v2/cases` | backend built; cases panel in the control room not built yet |
| **Live watchlist alerts** to volunteers' phones (SSE, optional Telegram) | `watchlist.py`, `/api/v2/alerts/stream` · `/v2/volunteer.html` | built |
| **Voice / text intake** (Bangla, Hindi, English → structured description; Gemini if `GEMINI_API_KEY` is set, else rules) | `/api/v2/intake/transcribe`, `/api/v2/intake/parse` | built |
| Control-room UI | `Frontend/v2/` at `/v2/` | built (search, results, trail); cases panel + polish pending |

**What a description can say.** The parser understands English, romanised Hindi and romanised Bengali:
- **Clothing:** colours and garment types, e.g. *laal / lal*, *neela / nil*, *peela / holud*, *kala / kalo*, *safed / shada*; t-shirt, jeans, frock, saree, kurta.
- **Physical:** age (number, "child", "elderly"), build and height.
- **Face and head:** beard (*dadi*), moustache (*mooch / gof*), glasses (*chashma / choshma*), cap (*topi / tupi*), turban, mask; hair colour, length and baldness (*kalo chul / kala baal*, *ganja / tak*).

The parsed attributes add a bonus on top of the OpenCLIP text↔image ranking.

## Quickstart

```powershell
py -3.11 -m venv venv              # Python 3.11, not 3.13/3.14 (insightface / onnxruntime wheels)
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env             # optional: GEMINI_API_KEY for voice intake, Telegram for alert push
python run.py
#   http://localhost:8000/v2/                  control room (v2)
#   http://<LAN-IP>:8000/v2/volunteer.html     volunteer phone page
#   http://localhost:8000/                     classic v0.5 UI
#   http://localhost:8000/docs                 API docs

# footage: put clips in CCTVS/ and copy docs/cameras.example.json to CCTVS/cameras.json
# (no footage? python scripts/make_synthetic_clip.py writes a test clip)
python -m src.backend.v2.indexer           # index once; unchanged files are skipped (--force to redo)
python scripts/check_models.py             # loads InsightFace, YOLO11n, OpenCLIP; prints device + latency
python scripts/smoke_v2.py [--photo face.jpg]   # health -> cameras -> index -> search, with timings
python -m pytest tests -q                  # pure-logic tests, no model downloads
```

## Status

### Known issues
Checked 13 Oct 2026 with `python -m pytest tests -q`: **30 passed, 10 failed, 5 errors.**
1. **Description search crashes for most queries.** `src/backend/v2/query_parser.py:182` uses `_AGE_PLAIN`, which is never defined. `parse()` raises `NameError` for any description without an explicit "N years". This also breaks text intake (7 tests).
2. **`google-genai` isn't installed** in the current venv, though it is in `requirements.txt` (3 intake tests). Fix: `pip install -r requirements.txt`.
3. **5 intake route tests error.** The venv has `httpx` 0.28.1 with `starlette` 0.27, whose `TestClient` passes `app=`, which httpx 0.28 removed. Fix: pin `httpx<0.28`, or upgrade fastapi/starlette.
4. **torch in the venv is CPU-only** (`2.6.0+cpu`), so YOLO and OpenCLIP run on CPU; InsightFace already uses CUDA. Fix: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`.

### Next steps
- Calibrate description search on multi-camera footage, then set `ATTR_BONUS_Z`, `LEAD_MIN_Z` and `TRAIL_MIN_Z`. `docs/CALIBRATION.md` has a partial one-camera result only.
- Build the cases panel in the control room (`Frontend/v2/cases.js`: list/create cases, "Search this case", index progress).
- Polish the UI for 1366×768, keyboard use and zero console errors.
- Benchmark v0.5 `POST /api/search` against v2 `POST /api/v2/search` on the same clips.

### Measured so far
On one test camera, CPU only, 10 Oct 2026 (details in `docs/CALIBRATION.md`):
- Indexing took 523 s for a 2,259-frame clip (753 frames sampled).
- Description search took 37–43 ms once OpenCLIP was loaded, and 5.8 s for the first query.

## Documentation
- [`docs/TRD.md`](docs/TRD.md): architecture, module map, API contract, pipelines, config.
- [`docs/PRD.md`](docs/PRD.md): problem, users, features, acceptance criteria, privacy commitments.
- [`docs/CALIBRATION.md`](docs/CALIBRATION.md): description-search measurements and how to calibrate.
- [`AGENTS.md`](AGENTS.md): rules for coding agents working on this repo.
- v0.5 docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/QUICK_START.md`](docs/QUICK_START.md), [`docs/QUICK_START_CCTV.md`](docs/QUICK_START_CCTV.md), [`DEPLOYMENT.md`](DEPLOYMENT.md), [`RECOGNITION_UPGRADE.md`](RECOGNITION_UPGRADE.md), [`DRISHTI_LIVE_PHONE_CAMERA_PLAN.md`](DRISHTI_LIVE_PHONE_CAMERA_PLAN.md).

---

## Drishti v0.5 (original version)

> The original v0.5 README follows unchanged. The classic UI and `POST /api/search` it describes still run alongside v2.

# DRISHTI - Lost Person Detection System

Find a lost person in CCTV footage from a single photo. Drishti scans every video in the `CCTVS/` folder with a modern face-recognition pipeline and tells you **which camera** saw the person, **when**, and **how confident** the match is and honestly says **"No Reliable Match"** when the evidence is weak, instead of showing the wrong person.

## Tech Stack

| Layer | Technology |
|---|---|
| Face detection + recognition | **InsightFace `buffalo_l`** (SCRFD detector + ArcFace embeddings) on **ONNX Runtime** |
| Video processing | **OpenCV** |
| Backend API | **FastAPI** + **Uvicorn** (Python) |
| Frontend | Single-page HTML/CSS/vanilla JS (served by the backend) |
| Fallback engine | Haar Cascade + ResNet18 (PyTorch) — legacy, optional |

## Features

- 🔍 **Modern face recognition** - InsightFace ArcFace embeddings with landmark-aligned faces
- 🛡️ **Open-set matching** - returns *Strong Match / Possible Match / No Reliable Match* (never a fake "98% confidence")
- 🎯 **Multi-frame confirmation** - a person must match across several frames (IoU tracking) before being reported
- 📹 **Per-camera results** - if the person appears in 4 feeds, you get a result card for each camera
- 🗂️ **Local CCTV feeds** - drop video files into `CCTVS/` and they automatically appear in the UI and get searched
- 🧹 **Quality gating** - blurry / tiny / badly-lit face crops are rejected before matching (with per-camera debug counts)
- 🎬 **Snapshots** - best frame + face crop saved for every confirmed match

**Pipeline:** `CCTV frames → SCRFD face detection → quality gate → ArcFace embedding → cosine similarity vs target → IoU track aggregation → open-set decision (per camera)`

## Project Structure
```
DRISHTI/
├── src/
│   ├── backend/
│   │   ├── app.py                   # FastAPI app & API routes
│   │   ├── search_service.py        # Search entry point (engine router + RTSP helpers)
│   │   ├── routes_camera.py         # Live-camera API (placeholder, future)
│   │   ├── vision/                  # Face recognition pipeline
│   │   │   ├── config.py            # ALL thresholds & tuning (env-overridable)
│   │   │   ├── insightface_engine.py# InsightFace buffalo_l (loads once)
│   │   │   ├── quality.py           # face quality gate
│   │   │   ├── matcher.py           # cosine similarity + open-set decision
│   │   │   ├── tracker.py           # IoU tracking, per-track aggregation
│   │   │   ├── pipeline.py          # search orchestration, per-camera results
│   │   │   ├── detector.py / recognizer.py  # backend seams (SCRFD/AdaFace-ready)
│   │   │   └── legacy_engine.py     # old Haar+ResNet18 fallback
│   │   └── camera/                  # rolling-buffer skeleton (future live feeds)
│   ├── config/settings.py           # server & directory settings
│   └── utils/
├── Frontend/                        # Web UI
│   ├── index.html
│   └── assets/
├── CCTVS/                           # ← put your CCTV videos here
├── data/
│   ├── uploads/                     # uploaded target photos
│   └── results/                     # match snapshots & crops
├── docs/
├── run.py                           # application entry point
└── requirements.txt
```


## System Interface

### Main UI Views

<div align="center">

| CCTV Interface | Search Results |
|---|---|
| ![CCTV UI](Frontend/assets/cctv_ui.png) | ![Result UI](Frontend/assets/result_ui.png) |

</div>

### System Process Flow

<div align="center">

|  |  |
|---|---|
| ![Process 1](Frontend/assets/process1.png) | ![Process 2](Frontend/assets/process2.png) |
| ![Process 3](Frontend/assets/process3.png) | ![Process 4](Frontend/assets/process4.png) |

</div>

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip
- ~1 GB free disk space (the InsightFace model pack, ~330 MB, downloads automatically on first search)

### Steps

1. **Clone / download the project**
   ```bash
   cd DRISHTI
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add CCTV videos**

   Put video files in the `CCTVS/` folder. Supported: `.mp4 .mov .avi .mkv .webm`

5. **Run**
   ```bash
   python run.py
   ```
   Open `http://localhost:8000` — the CCTV feeds appear automatically, one card per video.

> First search downloads the `buffalo_l` model to `~/.insightface/models` (internet needed once). The model loads a single time and is reused.

## Usage

1. Drop CCTV clips into `CCTVS/` (they show up as live feed cards in the UI).
2. Upload a clear photo of the lost person.
3. Click **Find Person**.
4. Results appear **per camera**: match strength (Strong/Possible), similarity score, confirmed frames, time range, and a snapshot. Cameras with a match are highlighted in the grid.
5. If the person can't be verified anywhere, the UI shows **"No Reliable Match Found"**.

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/search` | Upload a photo (`file`), search all CCTVS videos. Returns results directly. |
| `GET /api/search-results/{search_id}` | Re-fetch a previous search result |
| `GET /api/cameras` | List local CCTV feeds (`id`, `name`, `filename`, `video_url`) |
| `GET /api/cctv-video/{filename}` | Stream a video from `CCTVS/` (path-traversal safe) |
| `GET /api/snapshot/{filename}` | Match snapshot / face crop images |
| `GET /api/health` | Health check |
| `POST /api/camera/connect` etc. | Live-camera module (placeholder, not implemented yet) |

### `POST /api/search` response (shape)
```json
{
  "success": true,
  "search_id": "search_...",
  "results": {
    "decision": "match_found",
    "matches": [
      {
        "camera": "cctv1", "camera_name": "CCTV 1",
        "decision": "strong_match",
        "best_similarity": 0.63, "avg_similarity": 0.58, "matched_frames": 4,
        "timestamp_start": "00:00:02", "timestamp_end": "00:00:07",
        "best_frame": "/api/snapshot/...", "best_crop": "/api/snapshot/..."
      }
    ],
    "matches_by_camera": { "cctv1": ["..."] },
    "near_misses": ["cameras that were close to the threshold (debug)"],
    "per_camera_debug": { "cctv1": { "frames_processed": 41, "faces_detected": 502, "valid_faces": 357, "rejected": "...", "best_similarity_seen": 0.63 } }
  }
}
```
When nothing qualifies: `"decision": "no_reliable_match"` with an empty `matches` list.

## Configuration & Tuning

All recognition tuning lives in **`src/backend/vision/config.py`**. Every value can also be overridden with an environment variable (`DRISHTI_<NAME>`), no code edit needed.

| Setting | Default | Meaning |
|---|---|---|
| `MATCH_THRESHOLD` | `0.55` | Cosine floor for a **Strong Match** |
| `POSSIBLE_MATCH_THRESHOLD` | `0.45` | Floor for a **Possible Match** |
| `MIN_MATCH_FRAMES` | `2` | Frames a track must match before it counts |
| `MIN_FACE_SIZE` | `24` px | Smaller detected faces are ignored |
| `FRAME_SKIP` | `3` | Process every Nth frame (higher = faster, less recall) |
| `DET_SIZE` | `640` | Detector input size (`480`/`320` = much faster on CPU) |
| `RESULT_MODE` | `per_camera` | Best match from each camera (`global` = legacy top-N) |
| `MAX_RESULTS_PER_CAMERA` | `2` | Result cards allowed per camera |
| `RECOGNIZER_BACKEND` | `insightface` | `legacy` = old Haar+ResNet18 engine |

Examples:
```bash
# faster CPU demo
DRISHTI_DET_SIZE=480 DRISHTI_FRAME_SKIP=5 python run.py

# higher precision (fewer false positives, may miss distant faces)
DRISHTI_MATCH_THRESHOLD=0.60 python run.py
```

GPU: install `onnxruntime-gpu` and CUDA is used automatically (falls back to CPU).

## Match Decisions

| Decision | Meaning |
|---|---|
| **Strong Match** | Similarity ≥ 0.55 across multiple frames of the same track |
| **Possible Match** | Several frames above 0.45 — shown, but clearly weaker |
| **No Reliable Match** | Evidence too weak — by design, better than showing the wrong person |

Cosine similarity is shown as a score (e.g. `0.63`), never as a fake "confidence %".

## Troubleshooting

- **"No face detected in the uploaded photo"** — use a clearer, front-facing target photo.
- **Person missed** — try `DRISHTI_FRAME_SKIP=2` (denser sampling) or lower `DRISHTI_MATCH_THRESHOLD`; check `per_camera_debug` in the API response to see why faces were rejected.
- **Slow search** — CPU inference is the bottleneck: set `DRISHTI_DET_SIZE=480` (or `320`) and/or raise `DRISHTI_FRAME_SKIP`.
- **Feeds not showing** — confirm the files are in `CCTVS/` with a supported extension, then hard-refresh the browser.

## Limitations

- CPU-only search of a few short clips takes ~1–2 minutes (GPU is much faster).
- Accuracy depends on face size/visibility in the footage — distant or turned-away faces may be legitimately unmatchable.
- Live phone/IP-camera streaming is a planned module (API skeleton exists, not implemented yet).

## Credits

Built with:
- FastAPI
- InsightFace (ONNX Runtime)
- OpenCV
- Uvicorn

---

**Made By**: TeesMaarKhaCoders
