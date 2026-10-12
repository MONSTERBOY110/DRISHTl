# Drishti: Technical Requirements Document (TRD)

> Section 6 (API contract) is the agreement between backend and frontend. If you change it, update this file and `Frontend/v2/mock/*.json` in the same commit.
> Read with: [`PRD.md`](PRD.md) (what and why), the [`README`](../README.md#status) (current status, known issues), [`../AGENTS.md`](../AGENTS.md) (rules for agents).

---

## 1. Context

| Item | Value |
|---|---|
| Codebase | The v0.5 face-search core (`src/backend/vision/`, `src/backend/camera/`, classic UI `Frontend/index.html`) plus the v2 package (`src/backend/v2/`, `Frontend/v2/`) that builds on it |
| Reference machine | RTX 3050 6 GB laptop, Windows 11, Python 3.11 venv |

---

## 2. What the v0.5 core already does (reuse it; don't rebuild it)

| Path | What it does |
|---|---|
| `src/backend/app.py` | FastAPI app. `POST /api/search` (photo → face search over every video in `CCTVS/`), `GET /api/cameras`, `GET /api/cctv-video/{file}` (range-streaming, path-safe), `GET /api/snapshot/{file}`, `GET /api/health`. Mounts `Frontend/` at `/`. |
| `src/backend/vision/insightface_engine.py` | `buffalo_l` (SCRFD + ArcFace) lazy singleton, auto CUDA with Windows DLL preload. Only the detection and recognition modules are loaded. |
| `src/backend/vision/detector.py` | `extract_faces(frame_bgr) -> [{bbox:(x1,y1,x2,y2), kps, det_score, embedding(512)}]` |
| `src/backend/vision/quality.py` | `is_valid_face(face, frame, min_det=, min_size=, min_blur=) -> (ok, reason)` |
| `src/backend/vision/matcher.py` | `normalize`, `cosine`, `decide_track_match(similarities) -> {decision: strong_match / possible_match / no_reliable_match, ...}` |
| `src/backend/vision/config.py` | All face thresholds (`MATCH_THRESHOLD` 0.55, `POSSIBLE_MATCH_THRESHOLD` 0.45, `MIN_MATCH_FRAMES` 2, …), env-overridable with the `DRISHTI_` prefix. **Reuse these; don't fork them.** |
| `src/backend/vision/tracker.py` | IoU face tracker (per video) |
| `src/backend/vision/pipeline.py` | `search(target_path, videos, search_id)`. **Re-decodes every video on every search**, which takes ~1-2 min on CPU. v2 replaces this with index-once. |
| `src/backend/camera/*`, `routes_camera.py` | Phone/IP live camera: `POST /api/camera/connect`, MJPEG `/api/camera/stream/{id}`, status/list/disconnect. The worker indexes faces at 1 FPS into a 30 s `RollingBuffer`. `POST /api/search/live-buffer` searches it. |
| `Frontend/index.html` | Single page with inline CSS and JS: CCTV grid, photo upload, per-camera result cards, "Connect CCTV" modal |

### v0.5 quirks you must know

1. **Static mount shadows later routes.** `app.mount("/", StaticFiles(...))` sits at the end of `app.py`, so any router included after it is unreachable. v2 routers are registered *before* the mount by `register_v2(app)`.
2. **`Frontend/index.html` picks its backend from `config.js` (an old Render URL) whenever hostname ≠ localhost.** Opening it from a phone via LAN IP therefore calls Render. **v2 pages must always use same-origin `fetch('/api/v2/...')`.**
3. `Frontend/api.js`, `auth.js` and `script.js` call auth/targets endpoints that don't exist. They are legacy and unused by `index.html`, so don't build on them.
4. `src/config/settings.py` and `.env.example` hold stale values (MediaPipe, hog). The real tuning lives in `src/backend/vision/config.py` and `src/backend/v2/config.py`.
5. `RollingBuffer.recent(seconds)` returns `(entries, total, dropped)`. Each entry is `{timestamp(epoch s), frame_id, frame_jpeg(bytes), faces:[{bbox, det_score, embedding(normed)}]}`.

---

## 3. v2 architecture

```
 CCTVS/*.mp4 + CCTVS/cameras.json                    Phone / IP cameras (v0.5 live worker,
            │                                         1 FPS faces → RollingBuffer)
   [indexer]  once per file, skipped if unchanged                │
   per sampled frame:                                            │
     YOLO person detect + ByteTrack ......... persons.py         │
     InsightFace faces (v0.5 engine)                             │
     face → person association .............. associate.py      │
   per person-track:                                             │
     top-K crops → OpenCLIP image embedding .. appearance.py     │
     upper/lower clothing colour ............. appearance.py     │
     top-K face embeddings                                       │
            │                                                    ▼
            ▼                                        [watchlist] every 1 s:
   SQLite  data/v2/drishti_v2.db  (store.py)        buffer faces vs open cases
   cameras · tracks · cases · alerts                             │
            │                                                    ▼
   [search fusion]  ← photo and/or description        [alert bus] → SSE (+ optional Telegram)
            │          (query_parser.py parses it)               │
   /api/v2/search · /api/v2/cases                     /api/v2/alerts/stream
            │                                                    │
   Frontend/v2/index.html  control room               Frontend/v2/volunteer.html  phone
```

**Core idea:** decode video **once** at index time and store one record per *person track*, holding face embeddings (if a face was ever visible), an appearance embedding, and clothing colours. A search is then just numpy dot products over a few thousand vectors: **< 1 s instead of 1-2 min**. Because appearance is indexed, a person whose **face is never visible** can still be found from a text description.

---

## 4. Modules

Package: `src/backend/v2/`.

| Path | Purpose / public surface |
|---|---|
| `src/backend/v2/__init__.py` | `register_v2(app)`: includes every v2 router and starts background services (watchlist). |
| `src/backend/v2/config.py` | `DRISHTI_V2_*` env-overridable settings (see §8) |
| `src/backend/v2/store.py` | SQLite schema + DAO, embedding (de)serialisation, in-memory matrix cache with `invalidate()` |
| `src/backend/v2/indexer.py` | `index_all(force=False)`, `index_camera(cam)`, progress state; CLI `python -m src.backend.v2.indexer [--force]` |
| `src/backend/v2/search.py` | `search(photo_path=None, description=None, camera_ids=None) -> dict` (§6.4 shape), `build_trail(...)` |
| `src/backend/v2/routes_core.py` | `/api/v2/health`, `/cameras`, `/index/*`, `/search`, `/media/*` |
| `src/backend/v2/persons.py` | `PersonTracker()` → `.update(frame_bgr) -> [(track_id:int, bbox:(x1,y1,x2,y2), conf:float)]`; uses Ultralytics YOLO + ByteTrack, class 0 only |
| `src/backend/v2/appearance.py` | `embed_images(list[np.ndarray BGR]) -> (N,D) float32 L2-normed`, `embed_text(str) -> (D,)`, `color_attributes(person_crop_bgr) -> {"upper": "yellow", "lower": "blue"}` |
| `src/backend/v2/associate.py` | `assign_faces(faces, persons) -> {person_track_id: face_dict}` (face centre inside the upper 45 % of the person box; largest IoU wins) |
| `src/backend/v2/query_parser.py` | `parse(text) -> {"upper":[...], "lower":[...], "keywords":[...], "age_hint": str or None, "clip_text": str}`. Understands English plus romanised Hindi/Bengali clothing, face, hair and age words. |
| `src/backend/v2/intake.py` + `routes_intake.py` | Voice/text intake (§6.7): Gemini transcription + attribute normalisation when `GEMINI_API_KEY` is set, otherwise the `query_parser` rules |
| `src/backend/v2/cases.py` + `routes_cases.py` | Case CRUD, case photo → face embedding, `POST /cases/{id}/search` |
| `src/backend/v2/alerts.py` + `routes_alerts.py` | `publish(alert)` (thread-safe), SSE fan-out, ack; optional Telegram |
| `src/backend/v2/watchlist.py` | `start()` / `stop()` daemon thread: live buffers vs open cases → `alerts.publish` |
| `Frontend/v2/index.html`, `styles.css`, `api.js`, `search.js`, `trail.js`, `cases.js`, `mock/*.json` | Control-room UI. `api.js` has a mock switch (`?mock=1`). |
| `Frontend/v2/volunteer.html`, `volunteer.js` | Phone page: alert feed over SSE |
| `docs/cameras.example.json` → copy to `CCTVS/cameras.json` | Camera names, locations, clip start times, map positions |
| `scripts/*.py` | Smoke test, model check, person-tracking preview, synthetic clip, watchlist test |
| `tests/test_*.py` | pytest, pure logic (no model downloads in tests) |

### Rules
- `src/backend/app.py`: the only v2 hook is `register_v2(app)`, called *before* the static mount.
- `src/backend/v2/config.py`: settings are grouped in blocks by area (core, ML, intake, watchlist); add new ones to the matching block.
- v0.5 core (`vision/*`, `camera/*`): import from it rather than copying it. If you change it, check both the classic UI and v2.

---

## 5. Data model (SQLite `data/v2/drishti_v2.db`)

Embeddings are stored as `float32` bytes (`np.ndarray.tobytes()`). Shapes are recorded in the column comments.

```sql
CREATE TABLE IF NOT EXISTS cameras (
  id          TEXT PRIMARY KEY,         -- "cam1" (from cameras.json; else file stem)
  name        TEXT NOT NULL,            -- "CCTV 1"
  location    TEXT,                     -- "Main Gate"
  source      TEXT NOT NULL,            -- "file" | "live"
  file        TEXT,                     -- "cam1_gate.mp4" (in CCTVS/)
  start_time  TEXT,                     -- ISO-8601 wall clock when the clip starts, e.g. "2026-10-10T17:02:00+05:30"
  map_x REAL, map_y REAL,               -- 0..1 position on the venue map
  fps REAL, duration_s REAL,
  file_sig    TEXT,                     -- "size:mtime"; re-index only when this changes
  indexed_at  TEXT
);

CREATE TABLE IF NOT EXISTS tracks (     -- one row per person track per camera
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id    TEXT NOT NULL REFERENCES cameras(id),
  track_key    INTEGER NOT NULL,        -- ByteTrack id within that video
  t_start REAL, t_end REAL,             -- seconds from clip start
  n_frames     INTEGER,
  face_embs    BLOB,                    -- (K,512) float32, K<=TOPK_FACES, NULL if face never visible
  face_count   INTEGER DEFAULT 0,
  best_face_q  REAL,                    -- best face quality (det_score * size proxy)
  app_emb      BLOB NOT NULL,           -- (D,) float32 mean of top-K crop embeddings, re-normalised
  upper_color  TEXT, lower_color TEXT,
  best_crop    TEXT,                    -- filename in data/v2/crops/
  best_frame   TEXT,                    -- filename in data/v2/frames/ (bbox drawn)
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_cam ON tracks(camera_id);

CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,                  -- "case_0001"
  name TEXT NOT NULL, age INTEGER, description TEXT,
  photo_path TEXT,                      -- data/v2/cases/<id>.jpg
  face_emb BLOB,                        -- (512,) or NULL (no usable face in the photo)
  status TEXT NOT NULL DEFAULT 'open',  -- open | found | closed
  reporter_name TEXT, reporter_phone TEXT, last_seen_note TEXT,
  created_at TEXT, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY,                  -- "al_<epoch_ms>"
  case_id TEXT REFERENCES cases(id), camera_id TEXT,
  similarity REAL, decision TEXT,       -- strong_match | possible_match
  snapshot TEXT, crop TEXT,             -- filenames in data/v2/frames|crops
  status TEXT NOT NULL DEFAULT 'new',   -- new | ack | found | false_alarm
  created_at TEXT, acked_by TEXT, acked_at TEXT
);
```

Files: `data/v2/crops/`, `data/v2/frames/`, `data/v2/cases/`. All of `data/v2/` is git-ignored.

---

## 6. API contract (`/api/v2`): the frontend codes against this; mocks live in `Frontend/v2/mock/`

All responses carry `"success": true|false`. Errors: HTTP 4xx/5xx with `{"success": false, "message": "..."}`. Times: `t_*` = seconds from clip start; `wall_time` = ISO-8601 string or `null`.

### 6.1 `GET /api/v2/health`
```json
{"success": true, "version": "2.0",
 "index": {"cameras": 3, "tracks": 412, "last_indexed_at": "2026-10-10T19:40:00+05:30"},
 "models": {"face": "buffalo_l", "person": "yolo11n", "appearance": "ViT-B-32/laion2b_s34b_b79k"},
 "provider": "CUDAExecutionProvider"}
```

### 6.2 `GET /api/v2/cameras`
```json
{"success": true, "cameras": [
  {"id": "cam1", "name": "CCTV 1", "location": "Main Gate", "source": "file",
   "video_url": "/api/cctv-video/cam1_gate.mp4", "start_time": "2026-10-10T17:02:00+05:30",
   "map": {"x": 0.18, "y": 0.72}, "indexed": true, "tracks": 140}
]}
```
Live cameras connected through the classic UI's "Connect CCTV" modal appear with `"source": "live"` and `"video_url": "/api/camera/stream/<id>"`.

### 6.3 Indexing
- `POST /api/v2/index/rebuild`, body `{"force": false}` → `{"success": true, "state": "running"}` (runs in a background thread; 409 if already running)
- `GET /api/v2/index/status` →
```json
{"success": true, "state": "idle|running|done|error", "done": 1, "total": 3,
 "current_camera": "cam2", "tracks_indexed": 140, "elapsed_s": 41.2, "error": null}
```

### 6.4 `POST /api/v2/search` (multipart/form-data)
Fields: `photo` (file, optional), `description` (string, optional), `cameras` (comma-separated ids, optional), `case_id` (optional). **At least one of photo or description is required**, otherwise 400.

```json
{
  "success": true,
  "query": {"has_photo": true, "photo_face_found": true, "description": "yellow t-shirt, blue jeans, black cap",
            "parsed": {"upper": ["yellow"], "lower": ["blue"], "keywords": ["cap"], "age_hint": null}},
  "decision": "match_found",            // match_found | leads_only | no_reliable_match
  "verified": [ /* Sighting, face-verified, best first */ ],
  "leads":    [ /* Sighting, appearance-only, ranked, UNVERIFIED */ ],
  "trail":    [ /* TrailPoint, oldest → newest */ ],
  "last_seen": { /* TrailPoint */ },     // or null
  "timings": {"search_ms": 84, "tracks_scanned": 412},
  "message": "Verified in 2 cameras, 1 appearance lead."
}
```

**Sighting**
```json
{"sighting_id": "t_37", "track_id": 37, "camera_id": "cam2", "camera_name": "CCTV 2", "location": "Food Lane",
 "t_start": 41.3, "t_end": 47.9, "t_start_hms": "00:00:41", "wall_time": "2026-10-10T17:03:41+05:30",
 "face": {"decision": "strong_match", "best_similarity": 0.61, "matched_frames": 3},
 "appearance": {"score": 0.297, "z": 2.4, "rank": 1, "attribute_hits": ["upper:yellow", "lower:blue"]},
 "evidence": "face",                    // face | appearance
 "best_crop": "/api/v2/media/crops/cam2_t37.jpg",
 "best_frame": "/api/v2/media/frames/cam2_t37.jpg",
 "clip_url": "/api/cctv-video/cam2_corridor.mp4#t=41.3"}
```
`face` is `null` when the track never showed a usable face (or no photo was given). `appearance` is `null` when no description or photo-appearance signal was used.

**TrailPoint**
```json
{"camera_id": "cam2", "camera_name": "CCTV 2", "location": "Food Lane", "map": {"x": 0.5, "y": 0.4},
 "t_start": 41.3, "t_end": 47.9, "wall_time": "2026-10-10T17:03:41+05:30",
 "evidence": "appearance", "sighting_ids": ["t_37"]}
```

### 6.5 Cases
- `POST /api/v2/cases` (multipart): `name`*, `age`, `description`, `reporter_name`, `reporter_phone`, `last_seen_note`, `photo` → `{"success": true, "case": Case}`
- `GET /api/v2/cases` → `{"success": true, "cases": [Case]}` (newest first)
- `GET /api/v2/cases/{id}` → `{"success": true, "case": Case, "last_search": SearchResult|null, "alerts": [Alert]}`
- `PATCH /api/v2/cases/{id}`, body `{"status": "found"}` → `{"success": true, "case": Case}`
- `POST /api/v2/cases/{id}/search` → same body as 6.4, using the case's photo and description; result saved as `last_search`

**Case**
```json
{"id": "case_0001", "name": "Riya", "age": 6, "description": "yellow frock, red hair clip",
 "photo_url": "/api/v2/media/cases/case_0001.jpg", "has_face": true, "status": "open",
 "reporter": {"name": "Anita Das", "phone": "98xxxxxx10"}, "last_seen_note": "near pandal gate 2",
 "created_at": "2026-10-10T18:10:00+05:30", "updated_at": "2026-10-10T18:10:00+05:30"}
```

### 6.6 Alerts
- `GET /api/v2/alerts/stream`: **SSE**. Events: `event: alert` with `data: Alert`, plus `event: ping` every 15 s.
- `GET /api/v2/alerts?since=<iso>` → `{"success": true, "alerts": [Alert]}`
- `POST /api/v2/alerts/{id}/ack`, body `{"by": "Volunteer 3", "action": "ack|found|false_alarm"}` → `{"success": true, "alert": Alert}` (`found` also sets the case to `found`)
- `POST /api/v2/alerts/test`, optional body with alert fields → publishes a test alert (check that volunteer phones receive alerts)

**Alert**
```json
{"id": "al_1760101234567", "case_id": "case_0001", "case_name": "Riya", "camera_id": "live1",
 "camera_name": "Phone Cam", "location": "Stage Exit", "similarity": 0.63, "decision": "strong_match",
 "snapshot_url": "/api/v2/media/frames/al_1760101234567.jpg", "crop_url": "/api/v2/media/crops/al_1760101234567.jpg",
 "status": "new", "created_at": "2026-10-10T18:22:05+05:30"}
```

### 6.7 Intake
- `POST /api/v2/intake/transcribe` (multipart): `audio` (file, ≤ `INTAKE_MAX_AUDIO_BYTES`), `lang` (optional: `bn-IN|hi-IN|en-IN|auto`). Gemini transcribes the speech and normalises it into search attributes. Errors: 400 empty, 413 too large, 415 unsupported format, 503 when no `GEMINI_API_KEY` is set.
- `POST /api/v2/intake/parse` (JSON): `{"text": "...", "lang": "bn-IN|hi-IN|en-IN|auto"}`. Uses Gemini when a key is set, otherwise falls back to the `query_parser` rules. 400 on empty text.
- The exact response fields are defined in `src/backend/v2/intake.py` and covered by `tests/test_intake.py`.

### 6.8 Media
`GET /api/v2/media/{kind}/{filename}` with `kind ∈ {crops, frames, cases}`. Bare filename only (same traversal guards as v0.5 `/api/cctv-video`), `Cache-Control: no-store`.

---

## 7. Pipelines

### 7.1 Indexing
For each camera file whose `file_sig` changed (or `force`):
1. Open with OpenCV and read `fps`. Process every `DRISHTI_V2_FRAME_SKIP`-th frame (default 3).
2. `persons.update(frame)` → person boxes + track ids. Run `detector.extract_faces(frame)` (v0.5 engine) on the same frame, gate faces with `quality.is_valid_face`, then `associate.assign_faces(...)`.
3. Per track, keep a running record: `t_start/t_end`, `n_frames`, a heap of the top-`TOPK_CROPS` person crops by `area × sharpness`, and the top-`TOPK_FACES` face embeddings by `det_score × face size`.
4. At the end of the video, for every track with `n_frames ≥ MIN_TRACK_FRAMES`: `app_emb = normalise(mean(embed_images(top crops)))`, colours from the best crop, save best crop/frame JPEGs, insert the row.
5. Update `cameras.indexed_at`, `file_sig`, then `store.invalidate()`.

Progress lives in a module-level dict read by `/index/status`. Indexing runs in **one** background thread at a time.

### 7.2 Search fusion (scoring constants: see `docs/CALIBRATION.md`)
- **Face (photo given):** face-embed the photo with the v0.5 target gate (`TARGET_MIN_*`). For every track with `face_embs`: `sims = face_embs @ q`. `verdict = matcher.decide_track_match(list(sims))`. `strong_match` / `possible_match` → **verified**. Thresholds = v0.5 `vision/config.py`. Never invent new face thresholds.
- **Appearance (description given):** `t = embed_text(parsed.clip_text)`. `s = app_embs @ t`. `z = (s − mean(s)) / (std(s) + 1e-6)` over the scanned tracks. `+ ATTR_BONUS_Z` (default 0.75) per matching colour slot (upper/lower). Optionally `+ 0.5 × z_img` when a photo was given (photo vs crop image-image similarity; useful only for a same-day photo).
- **Leads** = tracks not already verified, sorted by `lead_z`, keeping `lead_z ≥ LEAD_MIN_Z` (default 1.0), at most `LEADS_MAX` (6). **Always labelled unverified in the UI.**
- `decision` = `match_found` if any verified, else `leads_only` if any leads, else `no_reliable_match`.

### 7.3 Trail
Points = all verified + leads with `lead_z ≥ TRAIL_MIN_Z` (default 1.5). `wall_time = camera.start_time + t_start` (`null` if the camera has no `start_time`; then order by camera list order and `t_start`). Merge consecutive points from the same camera less than 30 s apart. Sort by `wall_time`. `last_seen` = the final point.

### 7.4 Watchlist
A daemon thread, every `WATCH_INTERVAL_S` (1.0 s):
- `cases = open cases with face_emb`. For each live worker (`camera_manager.get(id)` for ids from `camera_manager.list_cameras()`): `entries, _, _ = worker.buffer.recent(WATCH_WINDOW_S)`.
- Per case: `hits` = number of entries whose best face similarity ≥ `MATCH_THRESHOLD`. If `hits ≥ WATCH_MIN_HITS` (2) and `(case, camera)` is outside `ALERT_COOLDOWN_S` (60 s) → decode the best entry's `frame_jpeg`, draw the bbox, save frame + crop, then `alerts.publish(...)`.
- Reads the v0.5 live-camera buffers read-only.

### 7.5 Alerts
`publish()` persists to SQLite, then fans out to SSE subscriber queues. It's called from a non-async thread, so it uses `loop.call_soon_threadsafe(queue.put_nowait, alert)` with the loop captured at startup. Optional Telegram: if `DRISHTI_V2_TELEGRAM_BOT_TOKEN` and `DRISHTI_V2_TELEGRAM_CHAT_ID` are set, `sendPhoto` with a caption. It must never block or crash the watchlist.

---

## 8. Config (`src/backend/v2/config.py`, env prefix `DRISHTI_V2_`)

| Name | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data/v2` | DB + media root |
| `FRAME_SKIP` | 3 | Index every Nth frame |
| `MIN_TRACK_FRAMES` | 3 | Drop shorter person tracks |
| `TOPK_CROPS` / `TOPK_FACES` | 5 / 5 | Per-track samples kept |
| `TRAIL_MERGE_S` / `LEAD_DEDUP_S` | 30 / 10 | Trail merge window; per-camera lead de-duplication window |
| `WARM_FACE_MODEL` | 1 | Load the face model at startup (set 0 on slow CPU-only boxes) |
| `PERSON_MODEL` | `models/yolo11n.pt` | Ultralytics weights path (keep it out of the repo root) |
| `PERSON_CONF` | 0.40 | Person detection confidence |
| `APPEARANCE_MODEL` / `APPEARANCE_PRETRAINED` | `ViT-B-32` / `laion2b_s34b_b79k` | OpenCLIP model |
| `ATTR_BONUS_Z`, `LEAD_MIN_Z`, `TRAIL_MIN_Z`, `LEADS_MAX` | 0.75, 1.0, 1.5, 6 | Search scoring; defaults, not yet calibrated (`docs/CALIBRATION.md`) |
| `LIVE_DESC_WINDOW_S` / `LIVE_DESC_MAX_FRAMES` | 12 / 8 | Description search over live buffers |
| `GEMINI_API_KEY` | unset | Voice/text intake; read from `GEMINI_API_KEY` or `DRISHTI_V2_GEMINI_API_KEY` |
| `GEMINI_TRANSCRIBE_MODEL` / `GEMINI_TEXT_MODEL` | `gemini-2.5-flash` | Intake models |
| `INTAKE_MAX_AUDIO_BYTES` / `INTAKE_TIMEOUT_S` | 10 MB / 15 | Intake limits |
| `WATCH_INTERVAL_S`, `WATCH_WINDOW_S`, `WATCH_MIN_HITS`, `ALERT_COOLDOWN_S` | 1.0, 10, 2, 60 | Watchlist |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | unset | Optional push |

`LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` still exist in `config.py` but nothing uses them; intake uses Gemini.

Secrets go in `.env` only (git-ignored). Never commit keys.

---

## 9. Dependencies and setup

- **Python 3.11** (the reference venv is 3.11.9). **Not 3.13/3.14**: `insightface` and `onnxruntime` wheels lag.
- v2 deps in `requirements.txt`: `ultralytics`, `open_clip_torch`, `lap` (ByteTrack), `google-genai` (intake), `pytest`, `httpx`. `torch` comes from the v0.5 list.
- Windows + `insightface`: `pip install insightface` compiles a C extension and needs **Microsoft C++ Build Tools** (Desktop C++ workload). If that blocks you, frontend work can run against mocks (`/v2/?mock=1`) or another machine's server over LAN.
- GPU (optional): `onnxruntime-gpu` + CUDA torch wheels. Everything also runs on CPU, more slowly.
- **Models download on first use; pre-download them before working offline:** `buffalo_l` (~330 MB → `~/.insightface/models`), `yolo11n.pt` (~6 MB → `models/`), OpenCLIP ViT-B-32 (~600 MB → HF cache). `scripts/check_models.py` loads all three.
- Run: `python run.py` → `http://localhost:8000/v2/` (control room), `http://<LAN-IP>:8000/v2/volunteer.html` (phone), `http://localhost:8000/` (classic v0.5 UI).

## 10. Performance budgets (to be measured, not assumed; record numbers in the README / `docs/CALIBRATION.md`)
| Metric | Target |
|---|---|
| Search latency after indexing (≤ 5k tracks) | < 1 s (v0.5: ~60-120 s CPU per search) |
| Index throughput on the reference GPU (RTX 3050) | ≥ real-time for 720p at FRAME_SKIP 3 |
| Live alert latency (person appears → phone shows alert) | < 5 s |
| False "verified" on distractors in test footage | 0 |

## 11. Testing
- `tests/` with pytest, **pure logic only** (parser, fusion maths, store round-trip with a temp DB, trail merge). No model downloads inside tests.
- `scripts/smoke_v2.py`: hits health → cameras → index status → search (photo / description / both) against a running server and prints timings.
- Every feature change ends with a manual check (smoke script or UI), and the result is reported honestly, including failures.

## 12. LAN and phone setup
- At a venue without reliable Wi-Fi, use a phone hotspot; the laptop and all phones join it.
- Find the laptop's IP with `ipconfig`. Allow Python through Windows Firewall on *Private* networks.
- The browser Notification API needs HTTPS, so on LAN HTTP the volunteer page uses an in-page alert plus `navigator.vibrate` and a sound after one "Enable alerts" tap. Telegram is the optional true push.

## 13. Risks and fallbacks
| Risk | Fallback |
|---|---|
| OpenCLIP too slow on CPU laptops | Index on a GPU machine; other machines use its server over LAN |
| Description search ranks poorly on real footage | Lean on colour attributes (deterministic, explainable); tune `ATTR_BONUS_Z` from `docs/CALIBRATION.md` measurements |
| ByteTrack fragments one person into many tracks | Leads de-duplicated per camera in a 10 s window; trail merge (§7.3) hides fragments |
| `insightface` won't install on a PC | Frontend work runs against mocks (`/v2/?mock=1`) or another machine's server over LAN |
| Live phone stream drops on a busy network | Dedicated hotspot (§12); recorded clips stay searchable |
