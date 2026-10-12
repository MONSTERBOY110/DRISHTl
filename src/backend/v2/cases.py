"""
Missing-person cases — DAO and business logic (M4, task D2).

Public surface
--------------
init_cases_db()                        → create the case_searches sidecar table
create_case(fields, photo_bytes)       → dict (Case shape, TRD §6.5)
get_case(case_id)                      → dict | None
list_cases()                           → list[dict], newest first
update_case_status(case_id, status)    → dict (updated Case)
get_last_search(case_id)               → dict | None (saved SearchResult)
save_last_search(case_id, result)      → None

Why a sidecar table for last_search: store.py's ``cases`` table has no column
for it, and track_id / sighting_id are regenerated on every re-index (CHECKLIST
gotcha 8), so the result must be a JSON snapshot — never a foreign-key reference.
"""

import logging
import shutil
import sqlite3
import time
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.backend.vision import detector, matcher, quality
from src.backend.v2 import config
from src.backend.v2.store import emb_to_blob, blob_to_faces, get_store, now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sidecar table (M4-owned — lives in the same DB but is not part of M1's SCHEMA)
# ---------------------------------------------------------------------------
_SIDECAR_DDL = """
CREATE TABLE IF NOT EXISTS case_searches (
    case_id    TEXT PRIMARY KEY REFERENCES cases(id),
    result_json TEXT NOT NULL,
    searched_at TEXT NOT NULL
);
"""


def init_cases_db() -> None:
    """Ensure the ``case_searches`` sidecar table exists.  Called once at startup."""
    with get_store().connect() as conn:
        conn.executescript(_SIDECAR_DDL)
    logger.info("v2 cases: case_searches table ready")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_case_id(conn: sqlite3.Connection) -> str:
    """Generate the next 'case_NNNN' id inside an open transaction."""
    n = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    return f"case_{int(n) + 1:04d}"


def _embed_photo(image_path: Path) -> Optional[np.ndarray]:
    """Return a normalised (512,) float32 ArcFace embedding, or None.

    Uses the lenient TARGET_MIN_* thresholds (same as search.embed_photo).
    Best face is chosen by highest det_score × face-area product.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"cases: could not read image at {image_path}")
        return None

    faces = detector.extract_faces(img)
    if not faces:
        logger.info(f"cases: no face detected in {image_path.name}")
        return None

    reject = quality.new_reject_counter()
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for face in faces:
        ok, reason = quality.is_valid_face(
            face, img,
            min_det=quality.config.TARGET_MIN_DETECTION_SCORE,
            min_size=quality.config.TARGET_MIN_FACE_SIZE,
            min_blur=quality.config.TARGET_MIN_BLUR_SCORE,
        )
        if not ok:
            if reason:
                reject[reason] = reject.get(reason, 0) + 1
            continue
        x1, y1, x2, y2 = face["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(face.get("det_score", 0.0)) * area
        if score > best_score:
            best_score = score
            best = face

    if best is None:
        logger.info(f"cases: all faces rejected for {image_path.name}: {reject}")
        return None

    emb = matcher.normalize(best.get("embedding"))
    if emb is None:
        logger.warning(f"cases: no embedding on best face in {image_path.name}")
        return None

    return emb.astype(np.float32)


def _save_photo(photo_bytes: bytes, case_id: str) -> Path:
    """Write raw bytes to CASES_DIR/<case_id>.jpg and return the path."""
    dest = config.CASES_DIR / f"{case_id}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(photo_bytes)
    return dest


def _format_case(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw DB row to the TRD §6.5 Case JSON shape."""
    photo_path: Optional[str] = row.get("photo_path")
    return {
        "id": row["id"],
        "name": row["name"],
        "age": row.get("age"),
        "description": row.get("description"),
        "photo_url": f"/api/v2/media/cases/{Path(photo_path).name}" if photo_path else None,
        "has_face": bool(row.get("face_emb")),   # non-None blob → True
        "status": row.get("status", "open"),
        "reporter": {
            "name": row.get("reporter_name"),
            "phone": row.get("reporter_phone"),
        },
        "last_seen_note": row.get("last_seen_note"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_case(
    name: str,
    age: Optional[int] = None,
    description: Optional[str] = None,
    reporter_name: Optional[str] = None,
    reporter_phone: Optional[str] = None,
    last_seen_note: Optional[str] = None,
    photo_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Insert a new missing-person case.  Returns the Case dict.

    Photo handling:
    - If *photo_bytes* is given, it is written to CASES_DIR and a face
      embedding is attempted with the lenient TARGET_MIN_* thresholds.
    - ``has_face`` = True only when a valid face was found in the photo.
    - If no photo or no face is found, ``face_emb`` stays NULL and description
      search is still fully functional.
    """
    store = get_store()
    ts = now_iso()

    # --- allocate ID + save photo atomically --------------------------------
    # We do the ID allocation inside the transaction so concurrent creates
    # can't collide: SELECT COUNT + INSERT happen in a single WAL write.
    photo_path_str: Optional[str] = None
    face_blob: Optional[bytes] = None

    with store.connect() as conn:
        case_id = _next_case_id(conn)

        # Save photo file (outside the DB lock is fine; we haven't inserted yet)
        if photo_bytes:
            photo_path = _save_photo(photo_bytes, case_id)
            photo_path_str = str(photo_path)

        conn.execute(
            """INSERT INTO cases
               (id, name, age, description, photo_path, face_emb,
                status, reporter_name, reporter_phone, last_seen_note,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id, name, age, description, photo_path_str, None,
                "open", reporter_name, reporter_phone, last_seen_note,
                ts, ts,
            ),
        )

    # --- extract face embedding (CPU-bound; done after the DB write) --------
    if photo_path_str:
        emb = _embed_photo(Path(photo_path_str))
        if emb is not None:
            face_blob = emb_to_blob(emb)
            with store.connect() as conn:
                conn.execute(
                    "UPDATE cases SET face_emb=?, updated_at=? WHERE id=?",
                    (face_blob, now_iso(), case_id),
                )
            logger.info(f"cases: face embedded for {case_id}")
        else:
            logger.info(f"cases: no usable face in photo for {case_id}")

    return get_case(case_id)  # type: ignore[return-value]


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    """Return the Case dict for *case_id*, or None if not found."""
    with get_store().connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if row is None:
        return None
    return _format_case(dict(row))


def list_cases() -> List[Dict[str, Any]]:
    """Return all cases, newest first."""
    with get_store().connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC"
        ).fetchall()
    return [_format_case(dict(r)) for r in rows]


def update_case_status(case_id: str, status: str) -> Optional[Dict[str, Any]]:
    """Set *status* on a case and return the updated Case dict.

    Allowed values: 'open', 'found', 'closed'.
    Returns None if the case does not exist.
    """
    allowed = {"open", "found", "closed"}
    if status not in allowed:
        raise ValueError(f"status must be one of {allowed}, got {status!r}")

    ts = now_iso()
    with get_store().connect() as conn:
        cur = conn.execute(
            "UPDATE cases SET status=?, updated_at=? WHERE id=?",
            (status, ts, case_id),
        )
        if cur.rowcount == 0:
            return None
    logger.info(f"cases: {case_id} status → {status}")
    return get_case(case_id)


def save_last_search(case_id: str, result: Dict[str, Any]) -> None:
    """Persist a search result snapshot for *case_id* (upsert)."""
    ts = now_iso()
    payload = json_dumps(result, default=str)
    with get_store().connect() as conn:
        conn.execute(
            """INSERT INTO case_searches (case_id, result_json, searched_at)
               VALUES (?,?,?)
               ON CONFLICT(case_id) DO UPDATE SET result_json=excluded.result_json,
                                                   searched_at=excluded.searched_at""",
            (case_id, payload, ts),
        )


def get_last_search(case_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent saved search result for *case_id*, or None."""
    with get_store().connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM case_searches WHERE case_id=?", (case_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        return json_loads(row["result_json"])
    except Exception:
        logger.warning(f"cases: could not parse last_search JSON for {case_id}")
        return None


def get_case_face_emb(case_id: str) -> Optional[np.ndarray]:
    """Return the (512,) float32 face embedding for *case_id*, or None.

    Used by watchlist.py (D4) to compare against live camera frames.
    """
    with get_store().connect() as conn:
        row = conn.execute(
            "SELECT face_emb FROM cases WHERE id=?", (case_id,)
        ).fetchone()
    if row is None or row["face_emb"] is None:
        return None
    embs = blob_to_faces(row["face_emb"])   # blob_to_faces handles (K,512) blobs
    if embs is None:
        return None
    # face_emb is stored as a single (512,) vector encoded by emb_to_blob,
    # which wraps a 1-D array; blob_to_faces reshapes to (-1, 512).
    if embs.ndim == 2 and embs.shape[0] >= 1:
        return embs[0]
    return None


def list_open_cases_with_face() -> List[Dict[str, Any]]:
    """Return all open cases that have a face embedding (for watchlist D4).

    Each item: {"id": str, "name": str, "face_emb": (512,) float32}
    """
    with get_store().connect() as conn:
        rows = conn.execute(
            "SELECT id, name, face_emb FROM cases WHERE status='open' AND face_emb IS NOT NULL"
        ).fetchall()
    result = []
    for r in rows:
        embs = blob_to_faces(r["face_emb"])
        if embs is None:
            continue
        emb = embs[0] if embs.ndim == 2 and len(embs) >= 1 else None
        if emb is not None:
            result.append({"id": r["id"], "name": r["name"], "face_emb": emb.astype(np.float32)})
    return result
