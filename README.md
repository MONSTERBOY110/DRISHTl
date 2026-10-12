# Drishti: find a lost person in the crowd from a photo or just a description

At a Durga Puja pandal a child gets separated from her family. Her mother has one old photo and can tell you what the child is wearing. Most CCTV never sees a face, because cameras are high and people face away.

Drishti indexes every camera's footage **once** into person tracks. Each track stores face embeddings when a face was visible, and clothing appearance always. A search is then a few vector products instead of re-scanning video:
- **A photo** gives face-verified sightings.
- **A description** like "yellow top, blue jeans" gives ranked *leads*, even when the face is never visible.
- The sightings are joined into a cross-camera *last seen* trail.

Leads are always labelled **unverified**. There is no fake "confidence %", and "No reliable match" is a valid answer.

## Features

| Feature | Where | Status |
|---|---|---|
| **Photo face search** over recorded CCTV: open-set decision, multi-frame confirmation, quality gate, per-camera results with snapshots | classic UI at `/` · `POST /api/search` | built |
| **Phone / IP cameras** as live feeds | `/api/camera/*` | built |
| **Index once, search instantly** (YOLO11n + ByteTrack people, InsightFace faces, OpenCLIP appearance) | `src/backend/v2/{indexer,store,search}.py` · `POST /api/v2/search` | built |
| **Search by description**, no face needed (OpenCLIP text↔image + clothing-colour attributes) | `query_parser.py`, `appearance.py` | built, but broken for most queries right now (Known issue 1) |
| **Cross-camera "last seen" trail** with clock time and venue map | `search.build_trail` · `/v2/` | built |
| **Missing-person cases**: register once with photo + description, search per case | `/api/v2/cases` | backend built; cases panel in the control room not built yet |
| **Live watchlist alerts** to volunteers' phones (SSE, optional Telegram) | `watchlist.py`, `/api/v2/alerts/stream` · `/v2/volunteer.html` | built |
| **Voice / text intake** in Bangla, Hindi or English, turned into a structured description (Gemini if `GEMINI_API_KEY` is set, else rules) | `/api/v2/intake/transcribe`, `/api/v2/intake/parse` | built |
| **Control-room UI** | `Frontend/v2/` at `/v2/` | built (search, results, trail); cases panel and polish pending |

**What a description can say.** The parser understands English, romanised Hindi and romanised Bengali:
- **Clothing:** colours and garment types, e.g. *laal / lal*, *neela / nil*, *peela / holud*, *kala / kalo*, *safed / shada*; t-shirt, jeans, frock, saree, kurta.
- **Physical:** age (number, "child", "elderly"), build and height.
- **Face and head:** beard (*dadi*), moustache (*mooch / gof*), glasses (*chashma / choshma*), cap (*topi / tupi*), turban, mask; hair colour, length and baldness (*kalo chul / kala baal*, *ganja / tak*).

The parsed attributes add a bonus on top of the OpenCLIP text↔image ranking.

## How it works

**Indexing (once per video, skipped if the file hasn't changed):**
`CCTV frames → YOLO11n + ByteTrack person tracks → InsightFace faces assigned to each person → OpenCLIP appearance embedding + upper/lower clothing colour → SQLite index`

**Search:**
- **Photo:** the photo's face embedding is compared with every indexed track that showed a face, using the open-set decision below. Matches are **verified**.
- **Description:** the text is embedded with OpenCLIP and scored against every track's appearance, plus a bonus per matching clothing colour. Top tracks are **leads (unverified)**.
- **Trail:** verified sightings and strong leads are ordered by clock time across cameras; the last point is "last seen".

**Classic photo search (v0.5, `POST /api/search`)** still runs alongside. It re-scans every video on each search:
`CCTV frames → SCRFD face detection → quality gate → ArcFace embedding → cosine similarity vs target → IoU track aggregation → open-set decision (per camera)`

### Match decisions

| Decision | Meaning |
|---|---|
| **Strong match** | Face similarity ≥ 0.55 across multiple frames of the same track |
| **Possible match** | Several frames above 0.45: shown, but clearly weaker |
| **Lead (unverified)** | Appearance-only result from a description; a human must check it |
| **No reliable match** | Evidence too weak. By design, this is better than showing the wrong person |

Similarity is shown as a score (e.g. `0.63`), never as a fake "confidence %".

## Tech stack

| Layer | Technology |
|---|---|
| Face detection + recognition | **InsightFace `buffalo_l`** (SCRFD detector + ArcFace embeddings) on **ONNX Runtime** |
| Person detection + tracking | **Ultralytics YOLO11n** + **ByteTrack** |
| Appearance (text ↔ image) | **OpenCLIP ViT-B-32** (LAION-2B weights) |
| Voice / text intake (optional) | **Google Gemini** via `google-genai` |
| Index store | **SQLite** |
| Video processing | **OpenCV** |
| Backend API | **FastAPI** + **Uvicorn** (Python) |
| Frontend | HTML/CSS/vanilla JS, no build step: classic UI, v2 control room, volunteer phone page |
| Fallback engine | Haar Cascade + ResNet18 (PyTorch); legacy, optional |

## Setup

**Prerequisites**
- **Python 3.11.** Not 3.13/3.14: the `insightface` and `onnxruntime` wheels lag behind.
- On Windows, `insightface` needs **Microsoft C++ Build Tools** (Desktop C++ workload).
- About 2 GB free disk for models, which download on first use: `buffalo_l` (~330 MB), OpenCLIP ViT-B-32 (~600 MB), `yolo11n.pt` (~6 MB).

```powershell
py -3.11 -m venv venv
venv\Scripts\activate                  # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # optional: GEMINI_API_KEY for voice intake, Telegram for alert push

# footage: put clips in CCTVS/ (.mp4 .mov .avi .mkv .webm)
# and copy docs/cameras.example.json to CCTVS/cameras.json (names, locations, start times, map positions)
# no footage? python scripts/make_synthetic_clip.py writes a test clip

python run.py
#   http://localhost:8000/v2/                  control room (v2)
#   http://<LAN-IP>:8000/v2/volunteer.html     volunteer phone page
#   http://localhost:8000/                     classic UI
#   http://localhost:8000/docs                 API docs

python -m src.backend.v2.indexer               # index once; unchanged files are skipped (--force to redo)
python scripts/check_models.py                 # loads InsightFace, YOLO11n, OpenCLIP; prints device + latency
python scripts/smoke_v2.py [--photo face.jpg]  # health -> cameras -> index -> search, with timings
python -m pytest tests -q                      # pure-logic tests, no model downloads
```

**GPU:** with `onnxruntime-gpu` installed, InsightFace uses CUDA automatically (falls back to CPU). YOLO and OpenCLIP need a CUDA build of torch (Known issue 4).

## Usage

**Control room (`/v2/`)**
1. Index the footage (CLI above, or `POST /api/v2/index/rebuild`).
2. Upload a photo, type or speak a description, or both. Optionally filter cameras.
3. Read **Verified** (face) and **Leads, unverified** (appearance) separately, then follow the trail map to the last-seen camera.
4. Volunteers open `/v2/volunteer.html` on their phones (same Wi-Fi or hotspot) and tap **Enable alerts**. Watchlist alerts arrive with a snapshot; they can mark **Ack**, **Found** or **False alarm**.

**Classic UI (`/`)**
1. Drop CCTV clips into `CCTVS/`; they show up as feed cards.
2. Upload a clear photo of the lost person and click **Find Person**.
3. Results appear per camera: match strength, similarity score, confirmed frames, time range and a snapshot. If the person can't be verified anywhere, the UI shows **"No Reliable Match Found"**.

## API

Full request and response shapes for v2: [`docs/TRD.md`](docs/TRD.md) §6. Interactive docs at `/docs`.

| Endpoint | Description |
|---|---|
| `GET /api/v2/health` | Version, index counts, models, execution provider |
| `GET /api/v2/cameras` | Recorded and live cameras with location, start time and map position |
| `POST /api/v2/index/rebuild` · `GET /api/v2/index/status` | Index footage in the background; progress |
| `POST /api/v2/search` | `photo` and/or `description` (+ `cameras`, `case_id`) → verified, leads, trail, last seen |
| `/api/v2/cases` | Create, list, read and update cases; `POST /api/v2/cases/{id}/search` |
| `GET /api/v2/alerts/stream` | Live alerts over SSE; `POST /api/v2/alerts/{id}/ack` to ack / found / false alarm |
| `POST /api/v2/intake/transcribe` · `POST /api/v2/intake/parse` | Voice or text → structured description |
| `GET /api/v2/media/{crops,frames,cases}/{file}` | Crops, frames and case photos |
| `POST /api/search` | Classic: upload a photo (`file`), search all `CCTVS/` videos |
| `GET /api/search-results/{search_id}` | Classic: re-fetch a previous search result |
| `GET /api/cameras` · `GET /api/cctv-video/{filename}` | Classic: list feeds; stream a video (path-traversal safe) |
| `GET /api/snapshot/{filename}` | Classic: match snapshots and face crops |
| `POST /api/camera/connect`, `GET /api/camera/stream/{id}` | Connect a phone/IP camera; MJPEG stream |
| `GET /api/health` | Health check |

<details>
<summary>Classic <code>POST /api/search</code> response shape</summary>

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
</details>

## Configuration

Face recognition settings live in **`src/backend/vision/config.py`**; override any of them with an environment variable `DRISHTI_<NAME>`, no code edit needed. v2 settings live in **`src/backend/v2/config.py`** with the prefix `DRISHTI_V2_` (full list in [`docs/TRD.md`](docs/TRD.md) §8).

| Setting | Default | Meaning |
|---|---|---|
| `MATCH_THRESHOLD` | `0.55` | Cosine floor for a **Strong match** |
| `POSSIBLE_MATCH_THRESHOLD` | `0.45` | Floor for a **Possible match** |
| `MIN_MATCH_FRAMES` | `2` | Frames a track must match before it counts |
| `MIN_FACE_SIZE` | `24` px | Smaller detected faces are ignored |
| `FRAME_SKIP` | `3` | Process every Nth frame (higher = faster, less recall) |
| `DET_SIZE` | `640` | Detector input size (`480`/`320` = much faster on CPU) |
| `RESULT_MODE` | `per_camera` | Best match from each camera (`global` = legacy top-N) |
| `MAX_RESULTS_PER_CAMERA` | `2` | Result cards allowed per camera |
| `RECOGNIZER_BACKEND` | `insightface` | `legacy` = old Haar+ResNet18 engine |

```bash
# faster on CPU
DRISHTI_DET_SIZE=480 DRISHTI_FRAME_SKIP=5 python run.py

# higher precision (fewer false positives, may miss distant faces)
DRISHTI_MATCH_THRESHOLD=0.60 python run.py
```

## Project structure
```
Drishti/
├── src/
│   ├── backend/
│   │   ├── app.py                 # FastAPI app, classic routes, calls register_v2(app)
│   │   ├── search_service.py      # classic search entry point
│   │   ├── routes_camera.py       # phone/IP camera API
│   │   ├── vision/                # face pipeline: InsightFace engine, quality gate, matcher, tracker, config
│   │   ├── camera/                # live camera workers + rolling buffer
│   │   └── v2/                    # index store, indexer, search, cases, alerts, watchlist, intake
│   ├── config/settings.py
│   └── utils/
├── Frontend/
│   ├── index.html                 # classic UI
│   ├── assets/
│   └── v2/                        # control room, volunteer page, mocks (?mock=1)
├── scripts/                       # smoke test, model check, synthetic clip, previews
├── tests/                         # pytest, pure logic
├── docs/                          # PRD, TRD, calibration, v0.5 architecture + quick starts
├── CCTVS/                         # your videos + cameras.json (not in git)
├── data/                          # uploads, results, v2 index + media (not in git)
├── run.py
└── requirements.txt
```

## Screenshots (classic UI)

<div align="center">

| CCTV Interface | Search Results |
|---|---|
| ![CCTV UI](Frontend/assets/cctv_ui.png) | ![Result UI](Frontend/assets/result_ui.png) |

|  |  |
|---|---|
| ![Process 1](Frontend/assets/process1.png) | ![Process 2](Frontend/assets/process2.png) |
| ![Process 3](Frontend/assets/process3.png) | ![Process 4](Frontend/assets/process4.png) |

</div>

## Status

### Known issues
Checked 13 Oct 2026 with `python -m pytest tests -q`: **30 passed, 10 failed, 5 errors.**
1. **Description search crashes for most queries.** `src/backend/v2/query_parser.py:182` uses `_AGE_PLAIN`, which is never defined. `parse()` raises `NameError` for any description without an explicit "N years". This also breaks text intake (7 tests).
2. **`google-genai` isn't installed** in the current venv, though it is in `requirements.txt` (3 intake tests). Fix: `pip install -r requirements.txt`.
3. **5 intake route tests error.** The venv has `httpx` 0.28.1 with `starlette` 0.27, whose `TestClient` passes `app=`, which httpx 0.28 removed. Fix: pin `httpx<0.28`, or upgrade fastapi/starlette.
4. **torch in the venv is CPU-only** (`2.6.0+cpu`), so YOLO and OpenCLIP run on CPU; InsightFace already uses CUDA. Fix: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`.

### Limitations
- Classic photo search re-scans every video, so a few short clips take about 1-2 minutes on CPU. v2 search runs on the index instead.
- Face matching depends on face size and visibility. Distant or turned-away faces may be legitimately unmatchable; description search exists for those cases.

### Next steps
- Calibrate description search on multi-camera footage, then set `ATTR_BONUS_Z`, `LEAD_MIN_Z` and `TRAIL_MIN_Z`. `docs/CALIBRATION.md` has a partial one-camera result only.
- Build the cases panel in the control room (`Frontend/v2/cases.js`: list/create cases, "Search this case", index progress).
- Polish the UI for 1366×768, keyboard use and zero console errors.
- Benchmark classic `POST /api/search` against `POST /api/v2/search` on the same clips.

### Measured so far
On one test camera, CPU only, 10 Oct 2026 (details in `docs/CALIBRATION.md`):
- Indexing took 523 s for a 2,259-frame clip (753 frames sampled).
- Description search took 37-43 ms once OpenCLIP was loaded, and 5.8 s for the first query.

## Troubleshooting
- **"No face detected in the uploaded photo":** use a clearer, front-facing photo.
- **Person missed:** try `DRISHTI_FRAME_SKIP=2` (denser sampling) or a lower `DRISHTI_MATCH_THRESHOLD`; check `per_camera_debug` in the classic API response to see why faces were rejected.
- **Slow classic search:** CPU inference is the bottleneck. Set `DRISHTI_DET_SIZE=480` (or `320`) and/or raise `DRISHTI_FRAME_SKIP`.
- **Feeds not showing:** confirm the files are in `CCTVS/` with a supported extension, then hard-refresh the browser.
- **A `/api/v2/...` route returns 404:** read the server log from startup; a v2 module that fails to import is logged as an error.

## Documentation
- [`docs/TRD.md`](docs/TRD.md): architecture, module map, API contract, pipelines, config.
- [`docs/PRD.md`](docs/PRD.md): problem, users, features, acceptance criteria, privacy commitments.
- [`docs/CALIBRATION.md`](docs/CALIBRATION.md): description-search measurements and how to calibrate.
- [`AGENTS.md`](AGENTS.md): rules for coding agents working on this repo.
- v0.5 docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/QUICK_START.md`](docs/QUICK_START.md), [`docs/QUICK_START_CCTV.md`](docs/QUICK_START_CCTV.md), [`DEPLOYMENT.md`](DEPLOYMENT.md), [`RECOGNITION_UPGRADE.md`](RECOGNITION_UPGRADE.md), [`DRISHTI_LIVE_PHONE_CAMERA_PLAN.md`](DRISHTI_LIVE_PHONE_CAMERA_PLAN.md).

---

**Made By**: TeesMaarKhaCoders
