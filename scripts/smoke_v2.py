"""
Smoke-test the v2 API against a RUNNING server (TRD §11, task A7).

Hits health → cameras → index status → search (description / photo / both) and
prints each result with its timing, so anyone can check "is the backbone up?"
in one command before touching the UI. Exits non-zero on the first failure.

Usage (server started with `python run.py`):
    python scripts/smoke_v2.py
    python scripts/smoke_v2.py --photo path/to/face.jpg --description "yellow top, blue jeans"
    python scripts/smoke_v2.py --rebuild          # also triggers an index run and waits for it
    python scripts/smoke_v2.py --base http://192.168.1.5:8000
"""

import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--photo", default=None, help="a photo with a face (optional)")
    ap.add_argument("--description", default="yellow top, blue jeans")
    ap.add_argument("--rebuild", action="store_true", help="POST /index/rebuild and wait for it")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base, timeout=120)
    failures = 0

    def step(name, fn):
        nonlocal failures
        t0 = time.perf_counter()
        try:
            out = fn()
            print(f"[ok]   {name:<34} {int((time.perf_counter() - t0) * 1000):>6} ms  {out}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name:<34} {e}")

    def health():
        j = c.get("/api/v2/health").raise_for_status().json()
        return f"index={j['index']} provider={j['provider']}"

    def cameras():
        j = c.get("/api/v2/cameras").raise_for_status().json()
        return ", ".join(f"{k['id']}({k['source']},{'idx' if k['indexed'] else 'no-idx'},{k['tracks']})" for k in j["cameras"]) or "none"

    def rebuild():
        r = c.post("/api/v2/index/rebuild", json={"force": False})
        if r.status_code == 409:
            return "already running"
        r.raise_for_status()
        for _ in range(600):
            st = c.get("/api/v2/index/status").json()
            if st["state"] in ("done", "error"):
                break
            time.sleep(1)
        if st["state"] != "done":
            raise RuntimeError(f"index ended in state {st['state']}: {st.get('error')}")
        return (f"{st['done']}/{st['total']} cameras, {st['tracks_indexed']} newly indexed tracks "
                f"(unchanged files skipped), {st['elapsed_s']} s")

    def status():
        return c.get("/api/v2/index/status").raise_for_status().json()["state"]

    def search(desc=None, photo=None):
        data = {"description": desc} if desc else {}
        files = {"photo": (photo, open(photo, "rb"), "image/jpeg")} if photo else None
        r = c.post("/api/v2/search", data=data, files=files)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            raise RuntimeError(j)
        for key in ("query", "decision", "verified", "leads", "trail", "last_seen", "timings", "message"):
            if key not in j:
                raise RuntimeError(f"missing key {key}")
        return (f"{j['decision']} verified={len(j['verified'])} leads={len(j['leads'])} trail={len(j['trail'])} "
                f"search_ms={j['timings']['search_ms']} scanned={j['timings']['tracks_scanned']}")

    step("GET /api/v2/health", health)
    step("GET /api/v2/cameras", cameras)
    if args.rebuild:
        step("POST /api/v2/index/rebuild (+wait)", rebuild)
    step("GET /api/v2/index/status", status)
    step("POST /api/v2/search (description)", lambda: search(desc=args.description))
    if args.photo:
        step("POST /api/v2/search (photo)", lambda: search(photo=args.photo))
        step("POST /api/v2/search (both)", lambda: search(desc=args.description, photo=args.photo))
    else:
        print("[skip] photo searches: pass --photo path.jpg")
    def neither():
        code = c.post("/api/v2/search", data={}).status_code
        if code != 400:
            raise RuntimeError(f"expected 400, got {code}")
        return "400 as expected"

    step("POST /api/v2/search (neither -> 400)", neither)   # ASCII only: Windows consoles are cp1252
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
