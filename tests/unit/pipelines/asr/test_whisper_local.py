"""Tests for the local Faster-Whisper provider."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from show_scribe.pipelines.asr.whisper_local import (
    TranscriptionOptions,
    build_whisper_transcriber,
)
from show_scribe.storage.paths import PathsConfig


class StubLoader:
    def __init__(self, model) -> None:
        self.model = model
        self.last_request: dict[str, object] | None = None

    def get(self, **kwargs):
        self.last_request = kwargs
        return self.model


class StubModel:
    def __init__(self, segments, info) -> None:
        self._segments = segments
        self._info = info
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **decode_options):
        self.calls.append({"audio_path": audio_path, "decode": decode_options})
        return iter(self._segments), self._info


def make_paths(tmp_path: Path) -> PathsConfig:
    paths = PathsConfig(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        cache_dir=tmp_path / "data" / "cache",
        temp_dir=tmp_path / "data" / "tmp",
        models_dir=tmp_path / "data" / "models",
        voice_bank_db=tmp_path / "data" / "voice_bank" / "voice.sqlite3",
        logs_dir=tmp_path / "logs",
    )
    paths.ensure_directories()
    return paths


def base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "providers": {
            "whisper": {
                "mode": "local",
                "model": "tiny",
                "compute_type": "int8",
                "device": "auto",
                "download_root": "./models",
                "beam_size": 5,
            }
        },
        "runtime": {
            "prefer_gpu": True,
            "device_priority": ["cuda", "mps", "cpu"],
            "max_workers": 2,
            "ffmpeg_path": None,
        },
        "transcription": {
            "language": None,
            "temperature": 0.0,
            "enable_word_timestamps": True,
            "initial_prompt": None,
            "word_timestamps_threshold": 0.5,
        },
        "paths": {
            "project_root": str(tmp_path),
        },
    }


def test_transcribe_returns_segments_with_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = base_config(tmp_path)
    paths = make_paths(tmp_path)

    segment = SimpleNamespace(
        id=0,
        start=0.0,
        end=1.0,
        text="Hello world",
        avg_logprob=-0.2,
        words=[
            SimpleNamespace(word="Hello", start=0.0, end=0.5, probability=0.9),
            SimpleNamespace(word="world", start=0.5, end=1.0, probability=0.6),
        ],
    )
    info = SimpleNamespace(language="en", duration=1.0, language_probability=0.95)
    model = StubModel([segment], info)
    loader = StubLoader(model)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"\x00\x00")

    transcriber = build_whisper_transcriber(config, paths, model_loader=loader)
    result = transcriber.transcribe(audio_path)

    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.text == "Hello world"
    assert pytest.approx(seg.confidence, 1e-6) == 0.75
    assert [word.word for word in seg.words] == ["Hello", "world"]
    assert result.metadata.language == "en"
    assert pytest.approx(result.metadata.detected_language_probability, 1e-6) == 0.95


def test_word_threshold_filters_low_probability_words(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    paths = make_paths(tmp_path)
    config["transcription"]["word_timestamps_threshold"] = 0.8  # type: ignore[index]

    segment = SimpleNamespace(
        id=1,
        start=2.0,
        end=4.0,
        text="Filtered segment",
        avg_logprob=-0.1,
        words=[
            SimpleNamespace(word="keep", start=2.0, end=3.0, probability=0.85),
            SimpleNamespace(word="drop", start=3.0, end=4.0, probability=0.4),
        ],
    )
    info = SimpleNamespace(language=None, duration=None, language_probability=None)
    model = StubModel([segment], info)
    loader = StubLoader(model)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"\x00\x00")

    transcriber = build_whisper_transcriber(config, paths, model_loader=loader)
    result = transcriber.transcribe(audio_path)

    words = result.segments[0].words
    assert len(words) == 1
    assert words[0].word == "keep"


def test_device_selection_prefers_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = base_config(tmp_path)
    paths = make_paths(tmp_path)

    segment = SimpleNamespace(
        id=0,
        start=0.0,
        end=0.5,
        text="",
        avg_logprob=-1.0,
        words=[],
    )
    model = StubModel(
        [segment],
        SimpleNamespace(language="en", duration=0.5, language_probability=1.0),
    )
    loader = StubLoader(model)

    monkeypatch.setattr("show_scribe.pipelines.asr.whisper_local._cuda_available", lambda: True)
    monkeypatch.setattr("show_scribe.pipelines.asr.whisper_local._mps_available", lambda: False)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"\x00\x00")

    transcriber = build_whisper_transcriber(config, paths, model_loader=loader)
    transcriber.transcribe(audio_path)

    assert loader.last_request is not None
    assert loader.last_request["device"] == "cuda"


def test_device_override_falls_back_to_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = base_config(tmp_path)
    config["providers"]["whisper"]["device"] = "cuda"  # type: ignore[index]
    paths = make_paths(tmp_path)

    segment = SimpleNamespace(
        id=0,
        start=0.0,
        end=0.5,
        text="",
        avg_logprob=-1.0,
        words=[],
    )
    model = StubModel(
        [segment],
        SimpleNamespace(language=None, duration=None, language_probability=None),
    )
    loader = StubLoader(model)

    monkeypatch.setattr("show_scribe.pipelines.asr.whisper_local._cuda_available", lambda: False)
    monkeypatch.setattr("show_scribe.pipelines.asr.whisper_local._mps_available", lambda: False)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")

    transcriber = build_whisper_transcriber(config, paths, model_loader=loader)
    transcriber.transcribe(audio_path)

    assert loader.last_request is not None
    assert loader.last_request["device"] == "cpu"


def test_language_override_passed_to_model(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    paths = make_paths(tmp_path)

    segment = SimpleNamespace(
        id=0,
        start=0.0,
        end=0.5,
        text="",
        avg_logprob=-1.0,
        words=[],
    )
    model = StubModel(
        [segment],
        SimpleNamespace(language=None, duration=None, language_probability=None),
    )
    loader = StubLoader(model)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")

    transcriber = build_whisper_transcriber(config, paths, model_loader=loader)
    transcriber.transcribe(audio_path, options=TranscriptionOptions(language="fr"))

    assert model.calls, "Expected model.transcribe to be invoked."
    decode_opts = model.calls[-1]["decode"]
    assert decode_opts["language"] == "fr"
