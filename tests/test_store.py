"""Pure-logic tests for src/backend/v2/store.py (temp DB, no models)."""

import sqlite3

import numpy as np
import pytest

from src.backend.v2 import store as store_mod
from src.backend.v2.store import FACE_DIM, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_db()
    return s


def _unit(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _track(camera_id, key, dim=64, faces=2, seed=0):
    return {
        "camera_id": camera_id, "track_key": key, "t_start": 1.0 * key, "t_end": 1.0 * key + 3,
        "n_frames": 5,
        "face_embs": np.stack([_unit(FACE_DIM, seed + i) for i in range(faces)]) if faces else None,
        "best_face_q": 0.9 if faces else None,
        "app_emb": _unit(dim, seed + 100),
        "upper_color": "yellow", "lower_color": "blue",
        "best_crop": f"{camera_id}_t{key}.jpg", "best_frame": f"{camera_id}_t{key}.jpg",
    }


def test_init_db_creates_all_tables(store):
    with sqlite3.connect(store.db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"cameras", "tracks", "cases", "alerts"} <= names


def test_camera_upsert_updates_only_provided_keys(store):
    store.upsert_camera({"id": "cam1", "name": "CCTV 1", "location": "Main Gate", "file": "cam1.mp4",
                         "file_sig": "10:20:app=hsv64", "map_x": 0.1, "map_y": 0.9})
    store.upsert_camera({"id": "cam1", "file_sig": "11:21:app=hsv64", "indexed_at": "2026-10-10T17:00:00+05:30"})
    cam = store.get_camera("cam1")
    assert cam["name"] == "CCTV 1" and cam["location"] == "Main Gate"   # untouched
    assert cam["file_sig"] == "11:21:app=hsv64" and cam["indexed_at"].startswith("2026-10-10")
    assert cam["source"] == "file"                                        # default on insert
    assert [c["id"] for c in store.list_cameras()] == ["cam1"]
    assert store.get_camera("nope") is None


def test_track_roundtrip_with_and_without_faces(store):
    store.upsert_camera({"id": "cam1"})
    t1 = _track("cam1", 1, faces=3, seed=1)
    t2 = _track("cam1", 2, faces=0, seed=2)
    id1 = store.insert_track(t1)
    id2 = store.insert_track(t2)

    r1 = store.get_track(id1)
    assert r1["face_embs"].shape == (3, FACE_DIM)
    assert np.allclose(r1["face_embs"], t1["face_embs"])
    assert np.allclose(r1["app_emb"], t1["app_emb"]) and r1["app_emb"].dtype == np.float32
    assert r1["face_count"] == 3 and r1["upper_color"] == "yellow" and r1["created_at"]

    r2 = store.get_track(id2)
    assert r2["face_embs"] is None and r2["face_count"] == 0
    assert store.count_tracks() == 2
    assert store.camera_track_counts() == {"cam1": 2}


def test_matrices_shape_filter_cache_and_invalidate(store):
    store.upsert_camera({"id": "cam1"})
    store.upsert_camera({"id": "cam2"})
    for cam, key in (("cam1", 1), ("cam1", 2), ("cam2", 3)):
        store.insert_track(_track(cam, key, seed=key))

    m = store.get_matrices(64)
    assert m["app_embs"].shape == (3, 64) and m["app_embs"].dtype == np.float32
    assert m["camera_ids"] == ["cam1", "cam1", "cam2"]
    assert len(m["face_embs"]) == 3 and m["face_embs"][0].shape == (2, FACE_DIM)
    assert m["t_start"].tolist() == [1.0, 2.0, 3.0]

    only1 = store.get_matrices(64, camera_ids=["cam1"])
    assert only1["app_embs"].shape == (2, 64) and only1["camera_ids"] == ["cam1", "cam1"]
    assert only1["track_ids"].tolist() == m["track_ids"][:2].tolist()

    store.insert_track(_track("cam2", 4, seed=4))
    assert store.get_matrices(64)["app_embs"].shape == (3, 64)      # still cached
    store.invalidate()
    assert store.get_matrices(64)["app_embs"].shape == (4, 64)      # reloaded


def test_matrices_skip_rows_with_other_appearance_dim(store):
    store.upsert_camera({"id": "cam1"})
    store.insert_track(_track("cam1", 1, dim=64, seed=1))
    store.insert_track(_track("cam1", 2, dim=512, seed=2))
    assert store.get_matrices(64)["app_embs"].shape == (1, 64)
    assert store.get_matrices(512)["app_embs"].shape == (1, 512)     # cache keyed by dim


def test_empty_matrices_have_right_shape(store):
    m = store.get_matrices(64)
    assert m["app_embs"].shape == (0, 64) and m["camera_ids"] == [] and m["face_embs"] == []


def test_delete_tracks_and_summary(store):
    store.upsert_camera({"id": "cam1", "indexed_at": "2026-10-10T17:00:00+05:30"})
    store.upsert_camera({"id": "cam2"})
    store.insert_track(_track("cam1", 1))
    store.insert_track(_track("cam2", 2))
    assert store.summary() == {"cameras": 1, "tracks": 2, "last_indexed_at": "2026-10-10T17:00:00+05:30"}
    assert store.delete_tracks("cam1") == 1
    assert store.count_tracks() == 1
    assert store.last_indexed_at() == "2026-10-10T17:00:00+05:30"


def test_blob_helpers():
    assert store_mod.emb_to_blob(None) is None
    assert store_mod.blob_to_faces(None) is None and store_mod.blob_to_faces(b"") is None
    faces = np.ones((2, FACE_DIM), dtype=np.float64)
    back = store_mod.blob_to_faces(store_mod.emb_to_blob(faces))
    assert back.shape == (2, FACE_DIM) and back.dtype == np.float32
