"""Tests for the hybrid Whisper transcriber fallback logic."""

from __future__ import annotations

from pathlib import Path

from show_scribe.pipelines.asr import (
    HybridWhisperTranscriber,
    TranscriptionResult,
    WhisperAPIError,
)
from show_scribe.pipelines.asr.whisper_api import WhisperAPITranscriber
from show_scribe.pipelines.asr.whisper_local import SegmentTranscription, TranscriptionMetadata


class StubLocal:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls: list[Path] = []

    def transcribe(self, audio_path: str | Path, *, options=None):
        self.calls.append(Path(audio_path))
        return self.result


class StubRemote(WhisperAPITranscriber):
    def __init__(self, result: TranscriptionResult, *, should_fail: bool = False) -> None:
        self.result = result
        self.should_fail = should_fail
        self.calls: list[Path] = []

    def transcribe(self, audio_path: str | Path, *, options=None, api_key: str | None = None):
        self.calls.append(Path(audio_path))
        if self.should_fail:
            raise WhisperAPIError("simulated failure")
        return self.result


def make_result(text: str) -> TranscriptionResult:
    segment = SegmentTranscription(
        segment_id=0,
        start=0.0,
        end=1.0,
        text=text,
        confidence=0.5,
        words=[],
    )
    metadata = TranscriptionMetadata(language="en", duration=1.0)
    return TranscriptionResult(segments=[segment], metadata=metadata)


def base_config(mode: str) -> dict[str, object]:
    return {
        "providers": {
            "whisper": {
                "mode": mode,
                "model": "tiny",
                "compute_type": "int8",
                "device": "cpu",
                "download_root": "./models",
                "beam_size": 1,
            },
            "whisper_api": {
                "api_base_url": "https://api.example.com/v1/audio/transcriptions",
                "timeout_seconds": 30,
                "max_retries": 0,
                "model": "whisper-1",
                "api_key_env": "TEST_KEY",
            },
        },
        "transcription": {
            "language": None,
            "temperature": 0.0,
            "enable_word_timestamps": True,
            "initial_prompt": None,
            "word_timestamps_threshold": 0.5,
        },
    }


def test_local_mode_uses_local_transcriber(tmp_path: Path) -> None:
    config = base_config("local")
    local_result = make_result("local")
    hybrid = HybridWhisperTranscriber(
        config,
        paths=None,
        local=StubLocal(local_result),
        remote=None,
    )

    output = hybrid.transcribe(tmp_path / "audio.wav")

    assert output.segments[0].text == "local"
    assert hybrid.remote is None


def test_api_success_returns_remote_result(tmp_path: Path) -> None:
    config = base_config("api")
    remote_result = make_result("remote")
    hybrid = HybridWhisperTranscriber(
        config,
        paths=None,
        local=StubLocal(make_result("local")),
        remote=StubRemote(remote_result),
    )

    output = hybrid.transcribe(tmp_path / "audio.wav")

    assert output.segments[0].text == "remote"


def test_api_failure_falls_back_to_local(tmp_path: Path) -> None:
    config = base_config("api")
    local_stub = StubLocal(make_result("local"))
    remote_stub = StubRemote(make_result("remote"), should_fail=True)

    hybrid = HybridWhisperTranscriber(
        config,
        paths=None,
        local=local_stub,
        remote=remote_stub,
    )

    output = hybrid.transcribe(tmp_path / "audio.wav")

    assert output.segments[0].text == "local"
    assert remote_stub.calls, "Expected remote to be attempted"
    assert local_stub.calls, "Expected fallback to local"
