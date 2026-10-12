"""
SQLite store for the v2 index (TRD §5): cameras, tracks, cases, alerts.

Why SQLite plus an in-memory matrix cache: a search must be a handful of numpy
dot products over every indexed track (< 1 s), not a per-row query. The rows
are the durable truth (``data/v2/drishti_v2.db``); ``get_matrices()`` stacks
the embeddings once and caches them until ``invalidate()`` is called after an
index run.

Embeddings are raw float32 bytes (``np.ndarray.tobytes()``). The face blob is
(K, 512) with K recovered from the byte length; the appearance blob is (D,)
where D depends on the appearance model in use (M1's stub ``hsv64`` = 64,
M2's OpenCLIP ViT-B-32 = 512). Rows whose D differs from the model in use are
skipped with a warning, never stacked, so a model swap can't crash a search
before the re-index finishes.

Thread safety: one short-lived connection per call under a re-entrant lock,
because the indexer thread, request threads and M4's watchlist thread all
write. M4 (cases, alerts) uses ``Store.connect()`` directly for its own rows.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)

FACE_DIM = 512  # ArcFace (buffalo_l) embedding size, fixed by the baseline engine

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  location    TEXT,
  source      TEXT NOT NULL,
  file        TEXT,
  start_time  TEXT,
  map_x REAL, map_y REAL,
  fps REAL, duration_s REAL,
  file_sig    TEXT,
  indexed_at  TEXT
);

CREATE TABLE IF NOT EXISTS tracks (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id    TEXT NOT NULL REFERENCES cameras(id),
  track_key    INTEGER NOT NULL,
  t_start REAL, t_end REAL,
  n_frames     INTEGER,
  face_embs    BLOB,
  face_count   INTEGER DEFAULT 0,
  best_face_q  REAL,
  app_emb      BLOB NOT NULL,
  upper_color  TEXT, lower_color TEXT,
  best_crop    TEXT,
  best_frame   TEXT,
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_cam ON tracks(camera_id);

CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL, age INTEGER, description TEXT,
  photo_path TEXT,
  face_emb BLOB,
  status TEXT NOT NULL DEFAULT 'open',
  reporter_name TEXT, reporter_phone TEXT, last_seen_note TEXT,
  created_at TEXT, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY,
  case_id TEXT REFERENCES cases(id), camera_id TEXT,
  similarity REAL, decision TEXT,
  snapshot TEXT, crop TEXT,
  status TEXT NOT NULL DEFAULT 'new',
  created_at TEXT, acked_by TEXT, acked_at TEXT
);
"""

CAMERA_COLS = ("id", "name", "location", "source", "file", "start_time", "map_x", "map_y",
               "fps", "duration_s", "file_sig", "indexed_at")
TRACK_COLS = ("camera_id", "track_key", "t_start", "t_end", "n_frames", "face_embs", "face_count",
              "best_face_q", "app_emb", "upper_color", "lower_color", "best_crop", "best_frame", "created_at")


def now_iso() -> str:
    """Local wall-clock time with offset, e.g. 2026-10-10T17:02:00+05:30."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Embedding (de)serialisation
# ---------------------------------------------------------------------------

def emb_to_blob(arr: Optional[np.ndarray]) -> Optional[bytes]:
    """float32 contiguous bytes, or None for a missing embedding."""
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def blob_to_faces(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """(K, 512) float32 or None. K is recovered from the byte length."""
    if not blob:
        return None
    flat = np.frombuffer(blob, dtype=np.float32)
    if flat.size == 0 or flat.size % FACE_DIM:
        return None
    return flat.reshape(-1, FACE_DIM)


def blob_to_app(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """(D,) float32 or None."""
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Store:
    """DAO over one SQLite file. Create one per DB path; ``get_store()`` holds the default."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_dim: Optional[int] = None

    # ---- connection -------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection; commits on success, rolls back on error."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        logger.info(f"v2 store ready: {self.db_path}")

    # ---- cameras ----------------------------------------------------------
    def upsert_camera(self, cam: Dict[str, Any]) -> None:
        """Insert a camera or update only the keys provided (others untouched)."""
        provided = {k: cam[k] for k in CAMERA_COLS if k in cam}
        if "id" not in provided:
            raise ValueError("camera needs an 'id'")
        row = dict(provided)
        row.setdefault("name", row["id"])      # NOT NULL defaults for a first insert only
        row.setdefault("source", "file")
        cols = list(row.keys())
        updates = ", ".join(f"{c}=excluded.{c}" for c in provided if c != "id")
        sql = (f"INSERT INTO cameras ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
               f"ON CONFLICT(id) DO " + (f"UPDATE SET {updates}" if updates else "NOTHING"))
        with self.connect() as conn:
            conn.execute(sql, [row[c] for c in cols])

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
        return dict(r) if r else None

    def list_cameras(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM cameras ORDER BY rowid").fetchall()
        return [dict(r) for r in rows]

    def camera_track_counts(self) -> Dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute("SELECT camera_id, COUNT(*) AS n FROM tracks GROUP BY camera_id").fetchall()
        return {r["camera_id"]: int(r["n"]) for r in rows}

    # ---- tracks -----------------------------------------------------------
    def delete_tracks(self, camera_id: str) -> int:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM tracks WHERE camera_id=?", (camera_id,))
        return int(cur.rowcount)

    def insert_track(self, track: Dict[str, Any]) -> int:
        """Insert one person track. ``face_embs`` (K,512)|None and ``app_emb`` (D,) are numpy arrays."""
        row = {k: track.get(k) for k in TRACK_COLS}
        if row["app_emb"] is None:
            raise ValueError("track needs an app_emb")
        faces = track.get("face_embs")
        row["face_embs"] = emb_to_blob(faces)
        row["face_count"] = int(track.get("face_count") or (0 if faces is None else len(faces)))
        row["app_emb"] = emb_to_blob(row["app_emb"])
        row["created_at"] = row["created_at"] or now_iso()
        cols = list(row.keys())
        with self.connect() as conn:
            cur = conn.execute(
                f"INSERT INTO tracks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
        return int(cur.lastrowid)

    def get_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            r = conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["face_embs"] = blob_to_faces(d["face_embs"])
        d["app_emb"] = blob_to_app(d["app_emb"])
        return d

    def count_tracks(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])

    def last_indexed_at(self) -> Optional[str]:
        with self.connect() as conn:
            v = conn.execute("SELECT MAX(indexed_at) FROM cameras").fetchone()[0]
        return v

    def summary(self) -> Dict[str, Any]:
        """The ``index`` block of GET /api/v2/health (TRD §6.1)."""
        with self.connect() as conn:
            cams = conn.execute("SELECT COUNT(*) FROM cameras WHERE indexed_at IS NOT NULL").fetchone()[0]
            tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            last = conn.execute("SELECT MAX(indexed_at) FROM cameras").fetchone()[0]
        return {"cameras": int(cams), "tracks": int(tracks), "last_indexed_at": last}

    # ---- matrix cache -----------------------------------------------------
    def invalidate(self) -> None:
        """Drop the cached matrices; call after every index run."""
        with self._lock:
            self._cache = None
            self._cache_dim = None

    def get_matrices(self, app_dim: int, camera_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Stacked per-track arrays for search, cached until ``invalidate()``.

        Keys: track_ids, track_keys, camera_ids (list), t_start, t_end, n_frames,
        app_embs (N, app_dim) float32, face_embs (list of (K,512)|None),
        face_count, best_face_q, upper, lower, best_crop, best_frame.
        ``camera_ids`` filters the cached set without reloading it.
        """
        with self._lock:
            if self._cache is None or self._cache_dim != app_dim:
                self._cache = self._load_matrices(app_dim)
                self._cache_dim = app_dim
            mats = self._cache
        if camera_ids is None:
            return mats
        wanted = set(camera_ids)
        mask = np.fromiter((c in wanted for c in mats["camera_ids"]), dtype=bool, count=len(mats["camera_ids"]))
        return _slice(mats, mask)

    def _load_matrices(self, app_dim: int) -> Dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, camera_id, track_key, t_start, t_end, n_frames, face_embs, face_count, "
                "best_face_q, app_emb, upper_color, lower_color, best_crop, best_frame "
                "FROM tracks ORDER BY id").fetchall()
        kept = []
        skipped = 0
        for r in rows:
            app = blob_to_app(r["app_emb"])
            if app is None or app.size != app_dim:
                skipped += 1
                continue
            kept.append((r, app))
        if skipped:
            logger.warning(
                f"v2 store: skipped {skipped} track(s) whose app_emb dim != {app_dim} "
                f"(indexed with another appearance model; re-index to use them)")
        n = len(kept)
        mats: Dict[str, Any] = {
            "track_ids": np.array([r["id"] for r, _ in kept], dtype=np.int64),
            "track_keys": np.array([r["track_key"] for r, _ in kept], dtype=np.int64),
            "camera_ids": [r["camera_id"] for r, _ in kept],
            "t_start": np.array([r["t_start"] or 0.0 for r, _ in kept], dtype=np.float32),
            "t_end": np.array([r["t_end"] or 0.0 for r, _ in kept], dtype=np.float32),
            "n_frames": np.array([r["n_frames"] or 0 for r, _ in kept], dtype=np.int64),
            "app_embs": (np.stack([a for _, a in kept]).astype(np.float32)
                         if n else np.zeros((0, app_dim), dtype=np.float32)),
            "face_embs": [blob_to_faces(r["face_embs"]) for r, _ in kept],
            "face_count": np.array([r["face_count"] or 0 for r, _ in kept], dtype=np.int64),
            "best_face_q": np.array([r["best_face_q"] or 0.0 for r, _ in kept], dtype=np.float32),
            "upper": [r["upper_color"] for r, _ in kept],
            "lower": [r["lower_color"] for r, _ in kept],
            "best_crop": [r["best_crop"] for r, _ in kept],
            "best_frame": [r["best_frame"] for r, _ in kept],
        }
        logger.info(f"v2 store: matrices loaded ({n} tracks, dim {app_dim})")
        return mats


def _slice(mats: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in mats.items():
        if isinstance(v, np.ndarray):
            out[k] = v[mask]
        else:
            out[k] = [x for x, keep in zip(v, mask) if keep]
    return out


# ---------------------------------------------------------------------------
# Default instance (config.DB_PATH)
# ---------------------------------------------------------------------------

_default: Optional[Store] = None
_default_lock = threading.Lock()


def get_store() -> Store:
    global _default
    with _default_lock:
        if _default is None:
            _default = Store(config.DB_PATH)
        return _default


def invalidate() -> None:
    """Module-level shortcut used by the indexer (TRD §7.1 step 5)."""
    get_store().invalidate()
