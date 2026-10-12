"""
Tests for voice intake and description normalization (M2 task B7).

Pure logic and mocked tests — no real network calls to Gemini or model downloads.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.v2 import config, intake
from src.backend.v2.routes_intake import router
from fastapi import FastAPI


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Audio Validation Unit Tests
# ---------------------------------------------------------------------------

def test_validate_audio_valid_mime():
    data = b"\x1aE\xdf\xa3" + b"x" * 100  # valid webm header + payload
    mime = intake.validate_audio(data, "audio/webm;codecs=opus")
    assert mime == "audio/webm"


def test_validate_audio_wav_magic_bytes():
    data = b"RIFF\x00\x00\x00\x00WAVE" + b"x" * 50
    mime = intake.validate_audio(data, "application/octet-stream")
    assert mime == "audio/wav"


def test_validate_audio_empty_raises_400():
    with pytest.raises(intake.IntakeError) as exc_info:
        intake.validate_audio(b"", "audio/webm")
    assert exc_info.value.status_code == 400


def test_validate_audio_too_short_raises_400():
    with pytest.raises(intake.IntakeError) as exc_info:
        intake.validate_audio(b"too short", "audio/webm")
    assert exc_info.value.status_code == 400


def test_validate_audio_unsupported_format_raises_415():
    data = b"x" * 200
    with pytest.raises(intake.IntakeError) as exc_info:
        intake.validate_audio(data, "application/pdf")
    assert exc_info.value.status_code == 415


def test_validate_audio_oversize_raises_413(monkeypatch):
    monkeypatch.setattr(config, "INTAKE_MAX_AUDIO_BYTES", 100)
    data = b"x" * 200
    with pytest.raises(intake.IntakeError) as exc_info:
        intake.validate_audio(data, "audio/webm")
    assert exc_info.value.status_code == 413


# ---------------------------------------------------------------------------
# Gemini Mocking & Transcription Tests
# ---------------------------------------------------------------------------

def test_transcribe_audio_missing_api_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(intake.UpstreamServiceError) as exc_info:
        intake.transcribe_audio(b"x" * 100, "audio/webm")
    assert "missing GEMINI_API_KEY" in str(exc_info.value)
    assert exc_info.value.status_code == 503


def test_transcribe_audio_successful_mock(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy-test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "bachha 6 saal ka laal t-shirt neela pant pehna hai"
    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr(intake, "get_genai_client", lambda: mock_client)

    result = intake.transcribe_audio(b"x" * 100, "audio/webm", lang="hi-IN")
    assert result == "bachha 6 saal ka laal t-shirt neela pant pehna hai"
    assert mock_client.models.generate_content.called


def test_transcribe_audio_handles_auth_error_without_leaking_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "sensitive-key-12345")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID: sensitive-key-12345 is bad")
    monkeypatch.setattr(intake, "get_genai_client", lambda: mock_client)

    with pytest.raises(intake.UpstreamServiceError) as exc_info:
        intake.transcribe_audio(b"x" * 100, "audio/webm")
    assert "sensitive-key-12345" not in str(exc_info.value)
    assert "authentication failed" in str(exc_info.value).lower()


def test_transcribe_audio_handles_rate_limit(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED: Rate limit exceeded")
    monkeypatch.setattr(intake, "get_genai_client", lambda: mock_client)

    with pytest.raises(intake.UpstreamServiceError) as exc_info:
        intake.transcribe_audio(b"x" * 100, "audio/webm")
    assert "rate limit" in str(exc_info.value).lower()


def test_normalize_description_valid_json(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "description_en": "child, 6 years old, red t-shirt, blue pants",
        "name": "Aarav",
        "age": 6,
    })
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(intake, "get_genai_client", lambda: mock_client)

    res = intake.normalize_description("bachha 6 saal ka laal t-shirt neela pant")
    assert res["description_en"] == "child, 6 years old, red t-shirt, blue pants"
    assert res["name"] == "Aarav"
    assert res["age"] == 6


def test_normalize_description_fallback_on_malformed_json(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not valid json at all!"
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(intake, "get_genai_client", lambda: mock_client)

    transcript = "yellow top blue jeans"
    res = intake.normalize_description(transcript)
    # Falls back gracefully to original transcript without crashing
    assert res["description_en"] == transcript
    assert res["name"] is None
    assert res["age"] is None


# ---------------------------------------------------------------------------
# End-to-End Route Tests
# ---------------------------------------------------------------------------

def test_route_transcribe_success(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy-key")

    fake_data = b"\x1aE\xdf\xa3" + b"x" * 200

    with patch.object(intake, "transcribe_audio", return_value="chhota bachha laal shirt neela pant"), \
         patch.object(intake, "normalize_description", return_value={
             "description_en": "child, red shirt, blue pants",
             "name": "Rohan",
             "age": 7,
         }):
        response = client.post(
            "/api/v2/intake/transcribe",
            files={"audio": ("speech.webm", fake_data, "audio/webm")},
            data={"lang": "hi-IN"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["engine"] == "gemini"
    assert data["transcript"] == "chhota bachha laal shirt neela pant"
    assert data["description_en"] == "child, red shirt, blue pants"
    assert "red" in data["parsed"]["upper"]
    assert "blue" in data["parsed"]["lower"]
    assert data["fields"]["name"] == "Rohan"
    assert data["fields"]["age"] == 7


def test_route_transcribe_empty_file_returns_400(client):
    response = client.post(
        "/api/v2/intake/transcribe",
        files={"audio": ("empty.webm", b"", "audio/webm")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "empty" in data["message"].lower()


def test_route_transcribe_unsupported_format_returns_415(client):
    response = client.post(
        "/api/v2/intake/transcribe",
        files={"audio": ("document.txt", b"x" * 100, "text/plain")},
    )
    assert response.status_code == 415
    data = response.json()
    assert data["success"] is False


def test_route_transcribe_missing_key_returns_503(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    fake_data = b"\x1aE\xdf\xa3" + b"x" * 200

    response = client.post(
        "/api/v2/intake/transcribe",
        files={"audio": ("speech.webm", fake_data, "audio/webm")},
    )
    assert response.status_code == 503
    data = response.json()
    assert data["success"] is False
    assert "not configured" in data["message"].lower()


def test_route_parse_text_rules_fallback(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    response = client.post(
        "/api/v2/intake/parse",
        json={"text": "chhele 8 bochor holud frock kalo chul"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["engine"] == "rules"
    assert "yellow" in data["parsed"]["upper"]
    assert "black hair" in data["parsed"]["facial_features"]
