"""Tests for the hosted Whisper API transcriber."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from show_scribe.pipelines.asr.whisper_api import (
    TranscriptionOptions,
    WhisperAPIError,
    build_whisper_api_transcriber,
)


class StubResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


class StubSession:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("No responses queued for StubSession")
        return self.responses.pop(0)


def base_config() -> dict[str, Any]:
    return {
        "providers": {
            "whisper_api": {
                "api_base_url": "https://api.example.com/v1/audio/transcriptions",
                "timeout_seconds": 30,
                "max_retries": 2,
                "model": "whisper-1",
                "api_key_env": "TEST_OPENAI_KEY",
            }
        },
        "transcription": {
            "language": None,
            "temperature": 0.0,
            "enable_word_timestamps": True,
            "initial_prompt": None,
            "word_timestamps_threshold": 0.5,
        },
    }


def write_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"\x00\x00")
    return audio_path


def test_transcribe_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = base_config()
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")

    response_payload = {
        "language": "en",
        "duration": 2.0,
        "language_probability": 0.92,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
                    {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.6},
                ],
            }
        ],
    }
    session = StubSession([StubResponse(200, response_payload)])
    transcriber = build_whisper_api_transcriber(config, session=session)

    audio_path = write_audio(tmp_path)
    result = transcriber.transcribe(audio_path)

    assert result.metadata.language == "en"
    assert pytest.approx(result.metadata.cost_usd, rel=0, abs=1e-4) == 0.0002
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.text == "Hello world"
    assert [word.word for word in segment.words] == ["Hello", "world"]
    assert session.calls, "Expected HTTP call to be recorded"
    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer secret"


def test_confidence_threshold_filters_words(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = base_config()
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")

    payload = {
        "segments": [
            {
                "id": 1,
                "start": 2.0,
                "end": 3.0,
                "text": "Filtered",
                "words": [
                    {"word": "keep", "start": 2.0, "end": 2.5, "confidence": 0.9},
                    {"word": "drop", "start": 2.5, "end": 3.0, "confidence": 0.4},
                ],
            }
        ]
    }
    session = StubSession([StubResponse(200, payload)])
    transcriber = build_whisper_api_transcriber(config, session=session)

    audio_path = write_audio(tmp_path)
    result = transcriber.transcribe(
        audio_path,
        options=TranscriptionOptions(word_confidence_threshold=0.8),
    )

    words = result.segments[0].words
    assert len(words) == 1
    assert words[0].word == "keep"
    assert pytest.approx(result.metadata.cost_usd, rel=0, abs=1e-4) == 0.0003


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = base_config()
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)

    session = StubSession([])
    transcriber = build_whisper_api_transcriber(config, session=session)
    audio_path = write_audio(tmp_path)

    with pytest.raises(WhisperAPIError):
        transcriber.transcribe(audio_path)


def test_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = base_config()
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")

    responses = [
        StubResponse(429, text="rate limited"),
        StubResponse(200, {"segments": []}),
    ]
    session = StubSession(responses)
    transcriber = build_whisper_api_transcriber(config, session=session)

    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    audio_path = write_audio(tmp_path)
    transcriber.transcribe(audio_path)

    assert len(session.calls) == 2


def test_exceeds_max_retries_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = base_config()
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret")

    responses = [StubResponse(500, text="server error") for _ in range(3)]
    session = StubSession(responses)
    transcriber = build_whisper_api_transcriber(config, session=session)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    audio_path = write_audio(tmp_path)
    with pytest.raises(WhisperAPIError):
        transcriber.transcribe(audio_path)
