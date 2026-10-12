# AGENTS.md: rules for coding agents on Drishti

Applies to any coding agent working in this repo. `CLAUDE.md` imports this file.

## 1. What this project is
**Drishti** finds a lost person (child, elderly) in CCTV and phone-camera footage at crowded events such as Kolkata's Durga Puja. It works from a **photo and/or a spoken or written description**.
- **v0.5 core** (`src/backend/vision/`, `src/backend/camera/`, classic UI `Frontend/index.html`): photo-based face search over recorded clips, plus a phone/IP live-camera module.
- **v2** (`src/backend/v2/`, `Frontend/v2/`):
  - index-once instant search
  - **description search when the face isn't visible**
  - a cross-camera "last seen" trail
  - missing-person cases with live watchlist alerts to volunteers' phones
  - voice/text intake

## 2. Read before you write code
1. `README.md`: features, setup, **current status and known issues**.
2. `docs/TRD.md`: architecture, module map (§4), **API contract (§6)**, config (§8).
3. `docs/PRD.md`: features and acceptance criteria.
4. `docs/CALIBRATION.md`: read it before changing any search threshold.

## 3. Hard rules
1. **Don't commit or push unless your human explicitly asks.** Finish the change, then print a commit message (§6) for your human to commit.
2. **Never rewrite pushed history.** No force push, no rebasing pushed commits.
3. **Never commit:** videos (`CCTVS/` is ignored), model weights (`*.pt`, `*.onnx`, `models/`), `data/`, `.env`, `venv/`, API keys.
4. **Honesty in the product:**
   - never show a fake "confidence %"
   - appearance-only results are labelled **"Lead (unverified)"**
   - "No reliable match" is a valid answer
   - every number in the README or docs must have been **measured**; say on what footage and hardware
5. **Privacy:**
   - search is case-scoped (a reported missing person); never add features that identify arbitrary people
   - everything runs locally, except the optional Gemini intake call and the optional Telegram alerts, which are both off unless their keys are set

## 4. Code layout and conventions
- **v0.5 face core** (`src/backend/vision/*`, `src/backend/camera/*`): v2 imports and reuses it: `vision.detector.extract_faces`, `vision.quality.is_valid_face`, `vision.matcher.decide_track_match`, and thresholds from `vision.config`. Don't fork these. If you change them, check both the classic UI (`/`) and v2 (`/v2/`).
- `src/backend/app.py`: `register_v2(app)` must stay **before** the static mount, or v2 routes become unreachable (TRD §2, quirk 1).
- Python:
  - a module docstring explaining the "why"
  - `logging.getLogger(__name__)`
  - type hints
  - settings in `src/backend/v2/config.py`, using `DRISHTI_V2_*` env overrides via the `_f/_i/_s` helpers
- FastAPI: one `APIRouter(prefix="/api/v2/...")` per area. Blocking work goes in `asyncio.to_thread`. Responses carry `success` + `message`.
- Changing the API? Update TRD §6 and `Frontend/v2/mock/*.json` in the same change.
- Frontend v2:
  - plain HTML/CSS/vanilla JS ES modules under `Frontend/v2/`, no build step
  - **same-origin fetch only** (`/api/v2/...`); never use `Frontend/config.js`, which points at an old Render URL
  - libraries from cdnjs or jsDelivr only
- Tests: `pytest`, pure logic, no model downloads (`tests/`).
- Keep diffs small: one feature or fix per commit.

## 5. Setup (Windows, PowerShell)
```powershell
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py                      # classic UI http://localhost:8000/ · v2 http://localhost:8000/v2/
python -m pytest tests -q
```
- Use **Python 3.11**, not 3.13/3.14. `insightface` on Windows needs Microsoft C++ Build Tools.
- Footage is not in git. Put clips in `CCTVS/` and copy `docs/cameras.example.json` to `CCTVS/cameras.json`.

## 6. When you finish a task
1. Run the relevant check (tests, `scripts/smoke_v2.py`, or the UI) and **report what you actually observed**, including failures.
2. If the change affects what works or what's broken, update the **Status** section of `README.md`.
3. Print a commit message in this format:

```
<type>(<scope>): <imperative summary, max ~72 chars>

- <what changed, file-level>
- <how it was verified>
```
- `type`: feat | fix | refactor | test | docs | chore | perf
- `scope`: core, store, index, search, persons, appearance, parser, associate, intake, ui, trail, cases, alerts, watchlist, volunteer, docs
- **No attribution trailers.** Don't add `Co-Authored-By:`, `Assisted-by:` or any agent, model or other person's name to commit messages.
