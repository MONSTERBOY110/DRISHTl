"""
Cases API router (M4, task D2).  TRD §6.5.

Endpoints
---------
POST   /api/v2/cases                    create a case (multipart)
GET    /api/v2/cases                    list all cases, newest first
GET    /api/v2/cases/{case_id}          get one case + last_search + alerts
PATCH  /api/v2/cases/{case_id}          update status
POST   /api/v2/cases/{case_id}/search   run a search using the case's photo + description

All blocking work runs in asyncio.to_thread() per the project convention.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.backend.v2 import cases as case_dao

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/cases", tags=["cases"])

# Initialise the case_searches sidecar table the first time this module is
# imported (register_v2 imports the router; store is already init'd by then).
case_dao.init_cases_db()


# ---------------------------------------------------------------------------
# POST /api/v2/cases  — create a missing-person case
# ---------------------------------------------------------------------------
@router.post("", status_code=201)
async def create_case(
    name: str = Form(...),
    age: Optional[int] = Form(None),
    description: Optional[str] = Form(None),
    reporter_name: Optional[str] = Form(None),
    reporter_phone: Optional[str] = Form(None),
    last_seen_note: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
):
    """Register a new missing-person case.

    *photo* is optional. When provided, the best face is extracted and stored as
    the search target for this case.  ``has_face`` in the response indicates
    whether a usable face was found.
    """
    photo_bytes: Optional[bytes] = None
    if photo is not None:
        photo_bytes = await photo.read()
        if not photo_bytes:
            photo_bytes = None

    try:
        case = await asyncio.to_thread(
            case_dao.create_case,
            name=name,
            age=age,
            description=description,
            reporter_name=reporter_name,
            reporter_phone=reporter_phone,
            last_seen_note=last_seen_note,
            photo_bytes=photo_bytes,
        )
    except Exception as exc:
        logger.exception("create_case failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"success": True, "case": case}


# ---------------------------------------------------------------------------
# GET /api/v2/cases  — list all cases, newest first
# ---------------------------------------------------------------------------
@router.get("")
async def list_cases():
    """Return all cases ordered newest first."""
    cases = await asyncio.to_thread(case_dao.list_cases)
    return {"success": True, "cases": cases}


# ---------------------------------------------------------------------------
# GET /api/v2/cases/{case_id}  — single case + last_search + alerts
# ---------------------------------------------------------------------------
@router.get("/{case_id}")
async def get_case(case_id: str):
    """Return one case, its last search result, and its alerts (newest 20)."""
    case = await asyncio.to_thread(case_dao.get_case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    last_search = await asyncio.to_thread(case_dao.get_last_search, case_id)

    # Pull alerts for this case from the alerts module if it exists.
    alerts: list = []
    try:
        from src.backend.v2 import alerts as alerts_mod
        alerts = await asyncio.to_thread(alerts_mod.list_alerts_for_case, case_id, limit=20)
    except (ImportError, AttributeError):
        pass  # alerts module not yet present (D3); return empty list

    return {"success": True, "case": case, "last_search": last_search, "alerts": alerts}


# ---------------------------------------------------------------------------
# PATCH /api/v2/cases/{case_id}  — update status
# ---------------------------------------------------------------------------
@router.patch("/{case_id}")
async def update_case_status(case_id: str, body: dict):
    """Update the case status.  Body: ``{"status": "open"|"found"|"closed"}``"""
    new_status = (body or {}).get("status")
    if new_status not in ("open", "found", "closed"):
        raise HTTPException(
            status_code=400,
            detail="'status' must be one of: open, found, closed",
        )

    updated = await asyncio.to_thread(case_dao.update_case_status, case_id, new_status)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    return {"success": True, "case": updated}


# ---------------------------------------------------------------------------
# POST /api/v2/cases/{case_id}/search  — run a search for this case
# ---------------------------------------------------------------------------
@router.post("/{case_id}/search")
async def search_case(case_id: str):
    """Run a search using the case's stored photo and description.

    The result is saved as ``last_search`` (upserted) and returned in the
    standard §6.4 shape.  No additional multipart fields are needed — the case
    already holds the photo path and description.
    """
    # Retrieve the raw DB row to access photo_path (not exposed in the formatted
    # Case dict to avoid leaking the server filesystem path).
    from src.backend.v2.store import get_store
    with get_store().connect() as conn:
        row = conn.execute(
            "SELECT id, photo_path, description FROM cases WHERE id=?", (case_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    photo_path: Optional[str] = row["photo_path"]
    description: Optional[str] = row["description"]

    if not photo_path and not description:
        raise HTTPException(
            status_code=400,
            detail="This case has neither a photo nor a description — nothing to search with.",
        )

    # Import here to avoid circular imports at module load time.
    from src.backend.v2.search import search

    try:
        result = await asyncio.to_thread(
            search,
            photo_path=photo_path,
            description=description,
        )
    except Exception as exc:
        logger.exception(f"search failed for case {case_id}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist the snapshot (upsert).
    await asyncio.to_thread(case_dao.save_last_search, case_id, result)

    return result
