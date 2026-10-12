"""
Voice and description intake service (M2 task B7).

Provides multilingual speech transcription (Bangla, Hindi, English, and code-mixed)
and search-attribute normalization via Google Gemini (official google-genai SDK),
with deterministic fallback to query_parser.parse().

Privacy & Ethics:
- Incoming audio is processed in memory and NEVER persisted to disk, repository,
  database, or logs.
- Sensitive personal identities (race, religion, caste, criminality, disability)
  are strictly never inferred.
- Only observable physical and clothing traits relevant to lost-person search are extracted.
"""

import json
import logging
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

from . import config, query_parser

logger = logging.getLogger(__name__)

_SUPPORTED_MIME_PREFIXES = (
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/aac",
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4",
    "audio/flac",
)

_client: Any = None
_client_lock = threading.Lock()


class IntakeError(Exception):
    """Base error for voice/intake processing."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UpstreamServiceError(IntakeError):
    """Upstream AI service failure (auth, timeout, quota)."""
    def __init__(self, message: str):
        super().__init__(message, status_code=503)


def get_genai_client() -> Any:
    """Return a lazy, reusable Google GenAI client, or None if GEMINI_API_KEY is not set."""
    global _client
    if not config.GEMINI_API_KEY:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            try:
                from google import genai
                _client = genai.Client(api_key=config.GEMINI_API_KEY)
                logger.info("GenAI client initialized successfully (lazy)")
            except Exception as e:
                logger.error("Failed to initialize GenAI client: %s", type(e).__name__)
                return None
    return _client


def validate_audio(data: bytes, content_type: Optional[str]) -> str:
    """Validate audio data size and MIME type. Returns normalized MIME type or raises IntakeError."""
    if not data or len(data) < 32:
        raise IntakeError("Audio data is empty or too short to process.", status_code=400)

    if len(data) > config.INTAKE_MAX_AUDIO_BYTES:
        max_mb = config.INTAKE_MAX_AUDIO_BYTES / (1024 * 1024)
        raise IntakeError(f"Audio size ({len(data)} bytes) exceeds the maximum allowed limit of {max_mb:.1f} MB.", status_code=413)

    raw_mime = (content_type or "").split(";")[0].strip().lower()

    # Detect by magic bytes if MIME is generic (e.g. application/octet-stream) or empty
    if not raw_mime or raw_mime in ("application/octet-stream", "binary/octet-stream"):
        if data.startswith(b"\x1aE\xdf\xa3"):
            raw_mime = "audio/webm"
        elif data.startswith(b"RIFF") and b"WAVE" in data[:12]:
            raw_mime = "audio/wav"
        elif data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            raw_mime = "audio/mp3"
        elif data.startswith(b"OggS"):
            raw_mime = "audio/ogg"
        elif len(data) >= 12 and data[4:8] == b"ftyp":
            raw_mime = "audio/mp4"

    if not any(raw_mime.startswith(prefix) for prefix in _SUPPORTED_MIME_PREFIXES):
        raise IntakeError(f"Unsupported audio format '{raw_mime}'. Please use webm, wav, mp3, ogg, or m4a.", status_code=415)

    return raw_mime


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fence if returned by model."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def transcribe_audio(audio_bytes: bytes, mime_type: str, lang: Optional[str] = None) -> str:
    """
    Transcribe audio bytes using Gemini speech understanding.
    Detects Bangla, Hindi, English, and mixed speech.
    Logs only operational metadata; never logs audio bytes, transcripts, or keys.
    """
    client = get_genai_client()
    if client is None:
        raise UpstreamServiceError("Voice transcription service is not configured (missing GEMINI_API_KEY).")

    from google.genai import types

    lang_hint = ""
    if lang and lang != "auto":
        lang_names = {
            "bn-IN": "Bengali (Bangla)",
            "hi-IN": "Hindi",
            "en-IN": "Indian English",
        }
        name = lang_names.get(lang, lang)
        lang_hint = f" The speaker is likely using {name} or a mix with English."

    prompt = (
        "You are an expert transcriber for emergency lost-person search operations (Drishti 2.0).\n"
        f"Transcribe the spoken audio accurately.{lang_hint}\n"
        "Guidelines:\n"
        "1. Faithfully capture what the speaker says in their original spoken language "
        "(Bengali, Hindi, English, or code-mixed Hinglish/Banglish).\n"
        "2. Do NOT summarize or invent missing details.\n"
        "3. Output ONLY the raw transcript text with no extra commentary or markdown formatting."
    )

    t0 = time.time()
    try:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=config.GEMINI_TRANSCRIBE_MODEL,
            contents=[audio_part, prompt],
        )
        elapsed = time.time() - t0
        logger.info(
            "Gemini transcription completed: bytes=%d mime=%s model=%s elapsed_ms=%.1f",
            len(audio_bytes), mime_type, config.GEMINI_TRANSCRIBE_MODEL, elapsed * 1000,
        )

        transcript = (response.text or "").strip()
        if not transcript:
            raise IntakeError("No speech could be recognized in the provided audio.", status_code=400)
        return transcript

    except IntakeError:
        raise
    except Exception as e:
        elapsed = time.time() - t0
        err_type = type(e).__name__
        logger.warning("Gemini transcription failed: type=%s elapsed_ms=%.1f", err_type, elapsed * 1000)
        err_msg = str(e).lower()
        if "api_key" in err_msg or "permission" in err_msg or "unauthenticated" in err_msg:
            raise UpstreamServiceError("Transcription authentication failed. Please verify API key configuration.")
        if "quota" in err_msg or "resource_exhausted" in err_msg or "rate" in err_msg:
            raise UpstreamServiceError("Transcription service rate limit exceeded. Please try again shortly.")
        if "deadline" in err_msg or "timeout" in err_msg:
            raise UpstreamServiceError("Transcription request timed out. Please try again with shorter audio.")
        raise UpstreamServiceError(f"Transcription service error ({err_type}).")


def normalize_description(transcript: str) -> Dict[str, Any]:
    """
    Normalize multilingual/colloquial transcript into concise English search attributes.
    Returns: {"description_en": str, "name": Optional[str], "age": Optional[int]}
    """
    client = get_genai_client()
    if client is None:
        # Fallback to identity transcript without calling LLM
        return {"description_en": transcript, "name": None, "age": None}

    prompt = (
        "You are an intake assistant for a lost-person search system (Drishti 2.0).\n"
        "Analyze the following report of a missing person (which may be in Bengali, Hindi, English, or mixed):\n\n"
        f"\"{transcript}\"\n\n"
        "Extract and normalize concise English search attributes describing the missing person.\n"
        "Focus strictly on observable visual features:\n"
        "- Clothing colors and types (upper garment, lower garment, footwear, accessories like cap, bag, spectacles)\n"
        "- Observable physical features (approximate age wording, gender if stated, build like slim/heavy, hair color/style, beard/mustache)\n"
        "- Name of the missing person (if explicitly stated)\n"
        "- Age number (if explicitly stated)\n\n"
        "STRICT SAFETY & ETHICS RULES:\n"
        "- Do NOT infer or mention race, religion, caste, disability, criminality, social status, or unspoken sensitive traits.\n"
        "- Do NOT hallucinate or assume unstated details.\n\n"
        "Respond with STRICTLY valid JSON with these exact keys:\n"
        "{\n"
        "  \"description_en\": \"concise English description (e.g. 'child, 6 years old, yellow top, blue jeans, black cap, short black hair, glasses')\",\n"
        "  \"name\": \"Name or null\",\n"
        "  \"age\": integer_or_null\n"
        "}"
    )

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=config.GEMINI_TEXT_MODEL,
            contents=prompt,
        )
        elapsed = time.time() - t0
        logger.info("Gemini description normalization completed: model=%s elapsed_ms=%.1f", config.GEMINI_TEXT_MODEL, elapsed * 1000)

        cleaned = _clean_json_markdown(response.text or "{}")
        data = json.loads(cleaned)
        desc_en = str(data.get("description_en") or transcript).strip()
        name = data.get("name")
        age = data.get("age")
        if isinstance(name, str):
            name = name.strip() or None
        else:
            name = None
        if not isinstance(age, (int, float)):
            age = None
        else:
            age = int(age)

        return {"description_en": desc_en, "name": name, "age": age}

    except Exception as e:
        logger.warning("Gemini description normalization failed (%s); using transcript directly", type(e).__name__)
        return {"description_en": transcript, "name": None, "age": None}


def process_audio_intake(audio_bytes: bytes, content_type: Optional[str], lang: Optional[str] = None) -> Dict[str, Any]:
    """
    End-to-end voice intake:
    1. Validate audio
    2. Transcribe using Gemini
    3. Normalize description to English
    4. Run query_parser.parse()
    """
    mime_type = validate_audio(audio_bytes, content_type)

    client = get_genai_client()
    if client is None:
        raise UpstreamServiceError("Voice transcription service is not configured (missing GEMINI_API_KEY).")

    transcript = transcribe_audio(audio_bytes, mime_type, lang=lang)
    normalized = normalize_description(transcript)
    desc_en = normalized["description_en"]
    parsed = query_parser.parse(desc_en)

    fields: Dict[str, Any] = {
        "name": normalized.get("name"),
        "age": normalized.get("age") or parsed.get("age_hint"),
    }

    return {
        "success": True,
        "engine": "gemini",
        "transcript": transcript,
        "description_en": desc_en,
        "parsed": parsed,
        "fields": fields,
        "message": "Description transcribed.",
    }


def process_text_intake(text: str, lang: Optional[str] = None) -> Dict[str, Any]:
    """
    Text-only intake (TRD §6.7 POST /api/v2/intake/parse):
    Normalizes free-text description and parses structured search attributes.
    """
    text = (text or "").strip()
    if not text:
        raise IntakeError("Description text is empty.", status_code=400)

    client = get_genai_client()
    if client is not None:
        normalized = normalize_description(text)
        desc_en = normalized["description_en"]
        engine = "gemini"
        fields = {
            "name": normalized.get("name"),
            "age": normalized.get("age"),
        }
    else:
        desc_en = text
        engine = "rules"
        fields = {"name": None, "age": None}

    parsed = query_parser.parse(desc_en)
    if not fields["age"] and parsed.get("age_hint"):
        fields["age"] = parsed["age_hint"]

    return {
        "success": True,
        "engine": engine,
        "description_en": desc_en,
        "parsed": parsed,
        "fields": fields,
        "message": "Description parsed successfully.",
    }
