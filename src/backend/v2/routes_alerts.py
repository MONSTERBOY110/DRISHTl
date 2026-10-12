"""
Alerts API router (M4, task D3).  TRD §6.6.

Endpoints
---------
GET  /api/v2/alerts/stream           SSE stream of new alerts + 15 s pings
GET  /api/v2/alerts?since=<iso>      list alerts, optional since filter
POST /api/v2/alerts/{id}/ack         ack / found / false_alarm

The SSE endpoint uses FastAPI's StreamingResponse with the async generator
from alerts.sse_stream().  The event loop is captured once at module import
time via alerts.capture_event_loop() so publish() (called from the watchlist
daemon thread) can push to subscriber queues safely.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.backend.v2 import alerts as alerts_mod

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/alerts", tags=["alerts"])

# Capture the event loop as soon as this module is imported (which happens
# inside register_v2, which runs before the static mount and before uvicorn
# starts serving requests — so the loop is already running by then).
alerts_mod.capture_event_loop()


# ---------------------------------------------------------------------------
# GET /api/v2/alerts/stream  — SSE
# ---------------------------------------------------------------------------
@router.get("/stream")
async def alerts_stream(request: Request):
    """Server-Sent Events stream.

    Events:
    - ``event: alert`` — a new alert (JSON Alert shape)
    - ``event: ping``  — keep-alive every 15 s

    Clients disconnect cleanly when the browser tab closes; the generator
    detects the cancellation via CancelledError and unregisters the queue.
    """
    async def _generate():
        async for chunk in alerts_mod.sse_stream():
            # Stop if the client disconnected
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
        },
    )


# ---------------------------------------------------------------------------
# GET /api/v2/alerts  — list with optional ?since= filter
# ---------------------------------------------------------------------------
@router.get("")
async def list_alerts(since: Optional[str] = None):
    """Return alerts newest-first.

    Query param ``since`` is an ISO-8601 timestamp; only alerts created after
    it are returned.  Useful for polling when SSE is not available.
    """
    result = await asyncio.to_thread(alerts_mod.list_alerts, since_iso=since)
    return {"success": True, "alerts": result}


# ---------------------------------------------------------------------------
# POST /api/v2/alerts/{alert_id}/ack  — ack / found / false_alarm
# ---------------------------------------------------------------------------
@router.post("/{alert_id}/ack")
async def ack_alert(alert_id: str, body: dict):
    """Acknowledge an alert.

    Body: ``{"by": "Volunteer 3", "action": "ack|found|false_alarm"}``

    ``found`` also sets the linked case status to ``found``.
    """
    by: str = (body or {}).get("by", "unknown")
    action: str = (body or {}).get("action", "")

    if action not in ("ack", "found", "false_alarm"):
        raise HTTPException(
            status_code=400,
            detail="'action' must be one of: ack, found, false_alarm",
        )

    try:
        updated = await asyncio.to_thread(alerts_mod.ack_alert, alert_id, by, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if updated is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    return {"success": True, "alert": updated}


# ---------------------------------------------------------------------------
# POST /api/v2/alerts/test  — trigger alert on running server
# ---------------------------------------------------------------------------
@router.post("/test")
async def trigger_test_alert(body: Optional[dict] = None):
    """Publish an alert on the running server (useful for demo, testing, and field drill)."""
    b = body or {}
    alert = await asyncio.to_thread(
        alerts_mod.publish,
        case_id=b.get("case_id", "case_0001"),
        case_name=b.get("case_name", "Riya"),
        camera_id=b.get("camera_id", "live1"),
        camera_name=b.get("camera_name", "Phone Cam"),
        location=b.get("location", "Stage Exit"),
        similarity=float(b.get("similarity", 0.68)),
        decision=b.get("decision", "strong_match"),
        snapshot=b.get("snapshot"),
        crop=b.get("crop"),
    )
    return {"success": True, "alert": alert}
