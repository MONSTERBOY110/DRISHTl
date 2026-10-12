"""Pure-logic tests for src/backend/v2/search.py: fusion maths and trail (temp DB, no models, no photo)."""

import numpy as np
import pytest

from src.backend.v2 import appearance, config, search
from src.backend.v2.store import Store


@pytest.fixture
def st(tmp_path, monkeypatch):
    # Pin the M2-calibrated constants so the tests don't depend on env overrides.
    monkeypatch.setattr(config, "ATTR_BONUS_Z", 0.75)
    monkeypatch.setattr(config, "LEAD_MIN_Z", 1.0)
    monkeypatch.setattr(config, "TRAIL_MIN_Z", 1.5)
    monkeypatch.setattr(config, "LEADS_MAX", 6)
    monkeypatch.setattr(config, "LEAD_DEDUP_S", 10.0)
    monkeypatch.setattr(config, "TRAIL_MERGE_S", 30.0)
    monkeypatch.setattr(appearance, "embed_text", lambda text: np.zeros(appearance.DIM, dtype=np.float32))
    s = Store(tmp_path / "t.db")
    s.init_db()
    s.upsert_camera({"id": "cam1", "name": "CCTV 1", "location": "Main Gate", "file": "cam1.mp4",
                     "start_time": "2026-10-10T17:02:00+05:30", "map_x": 0.15, "map_y": 0.75,
                     "indexed_at": "2026-10-10T18:00:00+05:30"})
    s.upsert_camera({"id": "cam2", "name": "CCTV 2", "location": "Food Lane", "file": "cam2.mp4",
                     "start_time": "2026-10-10T17:03:00+05:30", "map_x": 0.5, "map_y": 0.45,
                     "indexed_at": "2026-10-10T18:00:00+05:30"})
    return s


def _track(cam, key, t, upper, lower, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(appearance.DIM).astype(np.float32)
    return {"camera_id": cam, "track_key": key, "t_start": t, "t_end": t + 3.0, "n_frames": 5,
            "face_embs": None, "app_emb": v / np.linalg.norm(v), "upper_color": upper, "lower_color": lower,
            "best_crop": f"{cam}_t{key}.jpg", "best_frame": f"{cam}_t{key}.jpg"}


def test_description_leads_come_from_colour_attributes_only(st):
    st.insert_track(_track("cam1", 1, 5.0, "yellow", "blue", 1))     # both slots  -> z 1.5
    st.insert_track(_track("cam1", 2, 40.0, "yellow", "black", 2))   # upper only  -> z 0.75 (< LEAD_MIN_Z)
    st.insert_track(_track("cam2", 3, 7.0, "red", "black", 3))       # nothing     -> z 0

    r = search.search(description="yellow top, blue jeans", st=st)

    assert r["success"] and r["decision"] == "leads_only" and r["verified"] == []
    assert r["query"] == {"has_photo": False, "photo_face_found": False, "description": "yellow top, blue jeans",
                          "parsed": {"upper": ["yellow"], "lower": ["blue"], "keywords": ["top", "jeans"],
                                     "facial_features": [], "physical_features": [], "age_hint": None,
                                     "clip_text": "a photo of a person wearing yellow top and blue jeans"}}
    assert len(r["leads"]) == 1
    lead = r["leads"][0]
    assert lead["camera_id"] == "cam1" and lead["evidence"] == "appearance" and lead["face"] is None
    assert lead["appearance"] == {"score": 0.0, "z": 1.5, "rank": 1, "attribute_hits": ["upper:yellow", "lower:blue"]}
    assert lead["t_start_hms"] == "00:00:05" and lead["wall_time"] == "2026-10-10T17:02:05+05:30"
    assert lead["best_crop"] == "/api/v2/media/crops/cam1_t1.jpg" and lead["clip_url"] == "/api/cctv-video/cam1.mp4#t=5.0"
    assert r["trail"] == [{"camera_id": "cam1", "camera_name": "CCTV 1", "location": "Main Gate",
                           "map": {"x": 0.15, "y": 0.75}, "t_start": 5.0, "t_end": 8.0,
                           "wall_time": "2026-10-10T17:02:05+05:30", "evidence": "appearance", "sighting_ids": [lead["sighting_id"]]}]
    assert r["last_seen"] == r["trail"][0]
    assert r["timings"]["tracks_scanned"] == 3 and r["timings"]["search_ms"] >= 0
    assert "1 appearance lead(s), unverified" in r["message"]


def test_single_colour_description_can_surface_a_lead(st):
    st.insert_track(_track("cam1", 1, 5.0, "yellow", "black", 1))
    st.insert_track(_track("cam2", 2, 7.0, "red", "blue", 2))

    r = search.search(description="yellow shirt", st=st)

    assert r["decision"] == "leads_only"
    assert len(r["leads"]) == 1
    assert r["leads"][0]["camera_id"] == "cam1"
    assert r["leads"][0]["appearance"]["attribute_hits"] == ["upper:yellow"]
    assert r["last_seen"] is not None


def test_leads_deduplicated_per_camera_window_and_camera_filter(st):
    st.insert_track(_track("cam1", 1, 1.0, "yellow", "blue", 1))
    st.insert_track(_track("cam1", 2, 5.0, "yellow", "blue", 2))    # within 10 s of the first -> dropped
    st.insert_track(_track("cam1", 3, 30.0, "yellow", "blue", 3))   # separate sighting
    st.insert_track(_track("cam2", 4, 2.0, "yellow", "blue", 4))

    r = search.search(description="yellow top, blue jeans", st=st)
    assert [(l["camera_id"], l["t_start"]) for l in r["leads"]] == [("cam1", 1.0), ("cam1", 30.0), ("cam2", 2.0)]
    assert [l["appearance"]["rank"] for l in r["leads"]] == [1, 2, 3]
    # trail: cam1@1 and cam1@30 are 26 s apart (< 30 s merge window) -> one point; then cam2
    assert [(p["camera_id"], p["sighting_ids"]) for p in r["trail"]] == [
        ("cam1", [r["leads"][0]["sighting_id"], r["leads"][1]["sighting_id"]]), ("cam2", [r["leads"][2]["sighting_id"]])]
    assert r["last_seen"]["location"] == "Food Lane"

    r2 = search.search(description="yellow top, blue jeans", camera_ids=["cam2"], st=st)
    assert r2["timings"]["tracks_scanned"] == 1 and [l["camera_id"] for l in r2["leads"]] == ["cam2"]


def test_empty_index_is_honest(st):
    r = search.search(description="yellow top", st=st)
    assert r["decision"] == "no_reliable_match" and r["leads"] == [] and r["trail"] == [] and r["last_seen"] is None
    assert r["timings"]["tracks_scanned"] == 0 and "index/rebuild" in r["message"]


def test_build_trail_sorts_by_wall_time_and_merges(st):
    cams = {c["id"]: c for c in st.list_cameras()}
    def s(cam, t, ev, sid):
        return {"sighting_id": sid, "camera_id": cam, "camera_name": cams[cam]["name"], "location": cams[cam]["location"],
                "t_start": t, "t_end": t + 3.0, "wall_time": search._wall_time(cams[cam], t), "evidence": ev}
    # cam2 clock starts 60 s after cam1: cam2@10 = 17:03:10, cam1@5 = 17:02:05, cam1@20 = 17:02:20, cam1@100 = 17:03:40
    trail = search.build_trail([s("cam2", 10.0, "appearance", "b"), s("cam1", 5.0, "appearance", "a"),
                                s("cam1", 20.0, "face", "c"), s("cam1", 100.0, "appearance", "d")], cams)
    assert [(p["camera_id"], p["wall_time"], p["evidence"], p["sighting_ids"]) for p in trail] == [
        ("cam1", "2026-10-10T17:02:05+05:30", "face", ["a", "c"]),
        ("cam2", "2026-10-10T17:03:10+05:30", "appearance", ["b"]),
        ("cam1", "2026-10-10T17:03:40+05:30", "appearance", ["d"]),
    ]
    assert trail[0]["t_end"] == 23.0


def test_build_trail_without_clock_uses_camera_order():
    cams = {"camA": {"id": "camA", "name": "A"}, "camB": {"id": "camB", "name": "B"}}
    pts = [{"sighting_id": "2", "camera_id": "camB", "camera_name": "B", "location": None, "t_start": 1.0, "t_end": 2.0,
            "wall_time": None, "evidence": "appearance"},
           {"sighting_id": "1", "camera_id": "camA", "camera_name": "A", "location": None, "t_start": 9.0, "t_end": 10.0,
            "wall_time": None, "evidence": "appearance"}]
    trail = search.build_trail(pts, cams)
    assert [p["camera_id"] for p in trail] == ["camA", "camB"] and trail[0]["map"] is None


def test_wall_time_helper():
    assert search._wall_time({"start_time": "2026-10-10T17:02:00+05:30"}, 41.3) == "2026-10-10T17:02:41+05:30"
    assert search._wall_time({"start_time": None}, 1.0) is None
    assert search._wall_time({"start_time": "garbage"}, 1.0) is None
    assert search._hms(3661.9) == "01:01:01"
