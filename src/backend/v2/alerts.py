"""
Alert bus — persistence + SSE fan-out (M4, task D3).  TRD §7.5 / §6.6.

Public surface
--------------
init_alerts_db()                        → create the alerts table (already in M1 SCHEMA; this is a no-op guard)
capture_event_loop()                    → called once at FastAPI startup to capture the running loop
publish(alert_dict)                     → persist + fan-out to SSE subscribers (thread-safe, non-blocking)
list_alerts(since_iso=None)             → list[dict], newest first
list_alerts_for_case(case_id, limit)    → list[dict] for one case
get_alert(alert_id)                     → dict | None
ack_alert(alert_id, by, action)         → dict (updated alert); action ∈ {ack, found, false_alarm}

Design notes
------------
- publish() is called from the watchlist daemon thread (non-async). It uses
  loop.call_soon_threadsafe(queue.put_nowait, payload) so SSE consumers receive
  the alert without any thread-safety issues.
- Each SSE subscriber gets its own asyncio.Queue.  Subscribers register on
  connect and deregister on disconnect.
- The event loop is captured once at app startup (capture_event_loop()).
  If the loop isn't captured yet (server boot race), publish() falls back to
  DB-only and logs a warning — the volunteer can still poll /alerts.
- Optional Telegram: fires a sendPhoto request in a daemon thread so it never
  blocks or crashes the watchlist on network failure.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

from src.backend.v2 import config
from src.backend.v2.store import get_store, now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None          # captured at startup
_loop_lock = threading.Lock()

_subscribers: Set[asyncio.Queue] = set()                   # one queue per SSE client
_subscribers_lock = threading.Lock()


def capture_event_loop() -> None:
    """Capture the running asyncio event loop.  Call once from a FastAPI startup handler."""
    global _loop
    with _loop_lock:
        try:
            _loop = asyncio.get_event_loop()
            logger.info("alerts: event loop captured")
        except RuntimeError:
            logger.warning("alerts: no running event loop at capture time")


# ---------------------------------------------------------------------------
# Subscriber management (called from async context by the SSE endpoint)
# ---------------------------------------------------------------------------

def _subscribe() -> asyncio.Queue:
    global _loop
    try:
        cur_loop = asyncio.get_running_loop()
        with _loop_lock:
            _loop = cur_loop
    except RuntimeError:
        pass
    q: asyncio.Queue = asyncio.Queue()
    with _subscribers_lock:
        _subscribers.add(q)
    return q


def _unsubscribe(q: asyncio.Queue) -> None:
    with _subscribers_lock:
        _subscribers.discard(q)


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

def _format_alert(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw DB row to the TRD §6.6 Alert JSON shape."""
    snap = row.get("snapshot")
    crop = row.get("crop")
    return {
        "id": row["id"],
        "case_id": row.get("case_id"),
        "case_name": row.get("case_name"),        # denormalised at publish time
        "camera_id": row.get("camera_id"),
        "camera_name": row.get("camera_name"),    # denormalised
        "location": row.get("location"),          # denormalised
        "similarity": row.get("similarity"),
        "decision": row.get("decision"),
        "snapshot_url": f"/api/v2/media/frames/{snap}" if snap else None,
        "crop_url": f"/api/v2/media/crops/{crop}" if crop else None,
        "status": row.get("status", "new"),
        "created_at": row.get("created_at"),
        "acked_by": row.get("acked_by"),
        "acked_at": row.get("acked_at"),
    }


# ---------------------------------------------------------------------------
# publish()  — the core: DB insert + SSE fan-out + optional Telegram
# ---------------------------------------------------------------------------

def publish(
    *,
    case_id: str,
    case_name: str,
    camera_id: str,
    camera_name: str,
    location: Optional[str],
    similarity: float,
    decision: str,
    snapshot: Optional[str] = None,   # bare filename in data/v2/frames/
    crop: Optional[str] = None,       # bare filename in data/v2/crops/
) -> Dict[str, Any]:
    """Persist an alert and fan it out to all SSE subscribers.

    Safe to call from any thread.  Returns the formatted Alert dict.
    """
    alert_id = f"al_{int(time.time() * 1000)}"
    ts = now_iso()

    row: Dict[str, Any] = {
        "id": alert_id,
        "case_id": case_id,
        "case_name": case_name,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "location": location,
        "similarity": round(float(similarity), 4),
        "decision": decision,
        "snapshot": snapshot,
        "crop": crop,
        "status": "new",
        "created_at": ts,
    }

    # 1. Persist -------------------------------------------------------
    _db_insert(row)

    # 2. Format for wire -----------------------------------------------
    alert = _format_alert(row)

    # 3. SSE fan-out ---------------------------------------------------
    _fanout(alert)

    # 4. Optional Telegram push ----------------------------------------
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        threading.Thread(
            target=_send_telegram,
            args=(alert, snapshot),
            daemon=True,
            name="tg-alert",
        ).start()

    logger.info(f"alerts: published {alert_id} for case {case_id} on {camera_id} ({decision}, sim={similarity:.3f})")
    return alert


def _db_insert(row: Dict[str, Any]) -> None:
    cols = ["id", "case_id", "camera_id", "similarity", "decision",
            "snapshot", "crop", "status", "created_at"]
    vals = [row.get(c) for c in cols]
    with get_store().connect() as conn:
        conn.execute(
            f"INSERT INTO alerts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            vals,
        )


def _fanout(alert: Dict[str, Any]) -> None:
    """Push the alert to every SSE subscriber queue (thread-safe)."""
    global _loop
    with _loop_lock:
        loop = _loop
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
            with _loop_lock:
                _loop = loop
        except RuntimeError:
            pass
    if loop is None:
        logger.warning("alerts: event loop not captured yet; SSE fan-out skipped")
        return
    with _subscribers_lock:
        queues = list(_subscribers)
    for q in queues:
        try:
            loop.call_soon_threadsafe(q.put_nowait, alert)
        except Exception as exc:
            logger.warning(f"alerts: fan-out failed for one subscriber: {exc}")


def _send_telegram(alert: Dict[str, Any], snapshot_file: Optional[str]) -> None:
    """Fire-and-forget Telegram sendPhoto.  Runs in its own daemon thread."""
    import urllib.request
    import urllib.parse

    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    caption = (
        f"🚨 Alert: {alert.get('case_name', '?')} spotted\n"
        f"Camera: {alert.get('camera_name', '?')} ({alert.get('location', '?')})\n"
        f"Match: {alert.get('decision', '?')} (sim={alert.get('similarity', 0):.2f})\n"
        f"Case: {alert.get('case_id', '?')}"
    )

    try:
        if snapshot_file:
            snap_path = config.FRAMES_DIR / snapshot_file
            if snap_path.exists():
                # sendPhoto with file
                import http.client, mimetypes
                boundary = "---drishti_boundary"
                body_parts = []
                body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}".encode())
                body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}".encode())
                photo_data = snap_path.read_bytes()
                body_parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"alert.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
                    + photo_data
                )
                body_parts.append(f"--{boundary}--".encode())
                body = b"\r\n".join(body_parts)
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10):
                    pass
                return

        # Fallback: sendMessage only
        params = urllib.parse.urlencode({"chat_id": chat_id, "text": caption})
        url = f"https://api.telegram.org/bot{token}/sendMessage?{params}"
        with urllib.request.urlopen(url, timeout=10):
            pass

    except Exception as exc:
        logger.warning(f"alerts: Telegram push failed: {exc}")


# ---------------------------------------------------------------------------
# Read / ack
# ---------------------------------------------------------------------------

def list_alerts(since_iso: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Return alerts newest-first, optionally filtered by created_at > since_iso."""
    with get_store().connect() as conn:
        if since_iso:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
                (since_iso, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_format_alert(dict(r)) for r in rows]


def list_alerts_for_case(case_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return alerts for one case, newest first (used by GET /cases/{id})."""
    with get_store().connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE case_id=? ORDER BY created_at DESC LIMIT ?",
            (case_id, limit),
        ).fetchall()
    return [_format_alert(dict(r)) for r in rows]


def get_alert(alert_id: str) -> Optional[Dict[str, Any]]:
    with get_store().connect() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
    return _format_alert(dict(row)) if row else None


def ack_alert(alert_id: str, by: str, action: str) -> Optional[Dict[str, Any]]:
    """Update an alert's status.  action ∈ {ack, found, false_alarm}.

    'found' additionally sets the linked case to 'found'.
    Returns the updated Alert dict, or None if not found.
    """
    allowed = {"ack", "found", "false_alarm"}
    if action not in allowed:
        raise ValueError(f"action must be one of {allowed}, got {action!r}")

    # Map action → alert status
    status_map = {"ack": "ack", "found": "found", "false_alarm": "false_alarm"}
    new_status = status_map[action]
    ts = now_iso()

    with get_store().connect() as conn:
        cur = conn.execute(
            "UPDATE alerts SET status=?, acked_by=?, acked_at=? WHERE id=?",
            (new_status, by, ts, alert_id),
        )
        if cur.rowcount == 0:
            return None

        if action == "found":
            # Close the case as well
            row = conn.execute("SELECT case_id FROM alerts WHERE id=?", (alert_id,)).fetchone()
            if row and row["case_id"]:
                conn.execute(
                    "UPDATE cases SET status='found', updated_at=? WHERE id=?",
                    (ts, row["case_id"]),
                )
                logger.info(f"alerts: case {row['case_id']} marked found via alert {alert_id}")

    logger.info(f"alerts: {alert_id} acked by {by!r} — action={action}")
    return get_alert(alert_id)


# ---------------------------------------------------------------------------
# SSE helpers  (consumed by routes_alerts.py)
# ---------------------------------------------------------------------------

async def sse_stream():
    """Async generator yielding SSE-formatted strings for the /alerts/stream endpoint.

    Yields:
      - ``event: alert\\ndata: <json>\\n\\n``  when a new alert arrives
      - ``event: ping\\ndata: {}\\n\\n``        every 15 s (keep-alive)
    """
    import json

    q = _subscribe()
    try:
        while True:
            try:
                alert = await asyncio.wait_for(q.get(), timeout=15.0)
                yield f"event: alert\ndata: {json.dumps(alert)}\n\n"
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        _unsubscribe(q)
