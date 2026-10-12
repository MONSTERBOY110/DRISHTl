"""
FastAPI router for voice and text description intake (M2 task B7).

Endpoints:
- POST /api/v2/intake/transcribe (multipart/form-data)
- POST /api/v2/intake/parse (application/json, TRD §6.7)
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config, intake

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/intake", tags=["intake"])


class ParseRequest(BaseModel):
    text: str = Field(..., description="Free-text description of the missing person")
    lang: Optional[str] = Field(None, description="Optional language hint: bn-IN, hi-IN, en-IN, auto")


@router.post("/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(..., description="Recorded audio file"),
    lang: Optional[str] = Form(None, description="Optional language hint: bn-IN, hi-IN, en-IN, or auto"),
):
    """
    Transcribe recorded spoken description and normalize to search attributes.
    Content-Type: multipart/form-data
    """
    if not audio:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": false, "message": "Missing audio file in request."},
        )

    # Protect against oversize audio before buffering
    content_type = audio.content_type
    max_bytes = config.INTAKE_MAX_AUDIO_BYTES

    try:
        # Read up to max_bytes + 1 to detect oversize without unlimited memory consumption
        data = await audio.read(max_bytes + 1024)
        if len(data) > max_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"success": False, "message": f"Audio file exceeds {max_bytes / (1024 * 1024):.1f} MB limit."},
            )

        if not data:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": "Uploaded audio file is empty."},
            )

        result = await asyncio.to_thread(
            intake.process_audio_intake,
            data,
            content_type,
            lang,
        )
        return result

    except intake.IntakeError as e:
        logger.warning("Intake validation error: %s (status=%d)", e.message, e.status_code)
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.message},
        )
    except Exception as e:
        logger.error("Unhandled error in /intake/transcribe: %s", type(e).__name__, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "An internal error occurred while processing voice audio."},
        )
    finally:
        await audio.close()


@router.post("/parse")
async def parse_text_endpoint(payload: ParseRequest):
    """
    Parse a free-text description (TRD §6.7).
    Content-Type: application/json
    """
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "Description text cannot be empty."},
        )

    try:
        result = await asyncio.to_thread(
            intake.process_text_intake,
            text,
            payload.lang,
        )
        return result
    except intake.IntakeError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.message},
        )
    except Exception as e:
        logger.error("Unhandled error in /intake/parse: %s", type(e).__name__, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "An internal error occurred while parsing description."},
        )
