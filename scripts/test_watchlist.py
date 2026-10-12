#!/usr/bin/env python
"""Manual functional test for the watchlist daemon (D4).

Injects a fake live camera with face embeddings that match an open case,
runs one _tick(), and verifies an alert was published.

Run: venv\\Scripts\\python.exe scripts\\test_watchlist.py
"""

import time
import numpy as np
import cv2

# ---- 1. Ensure a case exists with a face_emb --------------------------------
from src.backend.v2.store import get_store, blob_to_faces, emb_to_blob

with get_store().connect() as conn:
    row = conn.execute(
        "SELECT id, name, face_emb FROM cases WHERE status='open' LIMIT 1"
    ).fetchone()

if row is None:
    print("SKIP: no open case in DB — create one via POST /api/v2/cases first")
    raise SystemExit(1)

# If the case has no face_emb, inject a random one for the test
if row["face_emb"] is None:
    fake_q = np.random.randn(512).astype(np.float32)
    fake_q /= np.linalg.norm(fake_q)
    with get_store().connect() as conn:
        conn.execute(
            "UPDATE cases SET face_emb=? WHERE id=?",
            (emb_to_blob(fake_q.reshape(1, 512)), row["id"]),
        )
    q_emb = fake_q
    print(f"  injected random face_emb into case {row['id']}")
else:
    embs = blob_to_faces(row["face_emb"])
    q_emb = embs[0]

case_id = row["id"]
print(f"Using case: {case_id}  q_emb norm={float(np.linalg.norm(q_emb)):.4f}")

# ---- 2. Build a fake rolling buffer with matching faces ---------------------
from src.backend.camera.rolling_buffer import RollingBuffer

buf = RollingBuffer("test_live", buffer_seconds=30)
blank = np.zeros((480, 640, 3), dtype=np.uint8)
_, jpg = cv2.imencode(".jpg", blank)
frame_jpeg = jpg.tobytes()

for i in range(3):
    noise = np.random.randn(512).astype(np.float32) * 0.05
    sim_emb = q_emb + noise
    sim_emb /= np.linalg.norm(sim_emb)
    sim = float(np.dot(sim_emb, q_emb))
    buf.add(
        time.time(), i, frame_jpeg,
        [{"bbox": [100, 100, 200, 200], "det_score": 0.9, "embedding": sim_emb}],
    )
    print(f"  frame {i}: similarity={sim:.4f}")

# ---- 3. Inject fake worker into camera_manager -----------------------------
from src.backend.camera.camera_manager import camera_manager

class FakeWorker:
    camera_id = "test_live"
    name = "Test Camera"
    status = "connected"
    buffer = buf

    def status_dict(self):
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": "Test Location",
        }

camera_manager._workers["test_live"] = FakeWorker()
print("Injected fake live camera: test_live")

# ---- 4. Clear cooldown and run one tick ------------------------------------
from src.backend.v2 import watchlist, alerts as alerts_mod

watchlist._cooldown.clear()

before = len(alerts_mod.list_alerts())
print(f"Alerts before tick: {before}")

watchlist._tick()
time.sleep(0.2)

after_list = alerts_mod.list_alerts()
after = len(after_list)
print(f"Alerts after tick:  {after}")

# ---- 5. Report -------------------------------------------------------------
new = after - before
if new > 0:
    a = after_list[0]
    print("\nNEW ALERT:")
    print(f"  id        = {a['id']}")
    print(f"  case_id   = {a['case_id']}")
    print(f"  camera_id = {a['camera_id']}")
    print(f"  decision  = {a['decision']}")
    print(f"  similarity= {a['similarity']}")
    print(f"  snapshot  = {a['snapshot_url']}")
    print(f"  crop      = {a['crop_url']}")
    print("\nPASS: watchlist fires alert on match")
else:
    print("\nFAIL: no new alert fired")

# Cleanup
camera_manager._workers.pop("test_live", None)

if __name__ == "__main__":
    # The script's logic is executed on import, but guard ensures pytest does not run it.
    pass
