"""Tests for the Pyannote diarization pipeline with mocked dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from src.show_scribe.pipelines.diarization import pyannote_pipeline as module
from src.show_scribe.pipelines.diarization.pyannote_pipeline import (
    PyannoteDiarizer,
    PyannoteNotAvailableError,
    build_pyannote_diarizer,
)

TEST_TOKEN = "hf_test_token"  # noqa: S105 - test fixture token placeholder


class DummySegment:
    """Minimal segment object mimicking pyannote.core.Segment."""

    __slots__ = ("end", "start")

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class DummyAnnotation:
    """Simplified annotation providing the subset of the API we rely on."""

    def __init__(self) -> None:
        self._entries: list[tuple[DummySegment, int, str]] = []
        self._details: dict[tuple[DummySegment, int], dict[str, Any]] = {}

    def add(
        self,
        start: float,
        end: float,
        label: str,
        *,
        track: int = 0,
        confidence: float | None = None,
    ) -> None:
        segment = DummySegment(start, end)
        self._entries.append((segment, track, label))
        if confidence is not None:
            self._details[(segment, track)] = {"confidence": confidence}

    def itertracks(self, yield_label: bool = False) -> Iterator[tuple[DummySegment, int, str]]:
        for segment, track, label in self._entries:
            if yield_label:
                yield segment, track, label
            else:
                yield segment, track

    def __getitem__(self, key: tuple[DummySegment, int]) -> dict[str, Any]:
        return self._details.get(key, {})

    def get_timeline(self) -> SimpleNamespace:
        start = self._entries[0][0].start
        end = self._entries[-1][0].end
        return SimpleNamespace(extent=lambda: SimpleNamespace(start=start, end=end))

    def labels(self) -> list[str]:
        return sorted({label for _, _, label in self._entries})


class DummyPipeline:
    """Fake pyannote pipeline that returns the prepared annotation."""

    def __init__(self, annotation: DummyAnnotation) -> None:
        self._annotation = annotation
        self._parameters = {
            "segmentation": {"threshold": 0.6, "onset": 0.3, "offset": 0.1},
            "clustering": {"threshold": 0.5, "overlap_rate": 0.4},
        }
        self.device = None

    @classmethod
    def from_pretrained(cls, model: str, use_auth_token: str) -> DummyPipeline:
        assert model == "pyannote/speaker-diarization@2.1"
        assert use_auth_token == TEST_TOKEN
        annotation = DummyAnnotation()
        annotation.add(0.0, 1.5, "SPEAKER_A", confidence=0.8)
        annotation.add(1.5, 3.0, "SPEAKER_B", confidence=0.6)
        return cls(annotation)

    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def instantiate(self, params: dict[str, Any]) -> None:
        self._parameters = params

    def to(self, device: Any) -> None:
        self.device = device

    def __call__(self, *_, **__) -> DummyAnnotation:
        return self._annotation


@pytest.fixture(autouse=True)
def _patch_pyannote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOW_SCRIBE_PYANNOTE_TOKEN", TEST_TOKEN)
    monkeypatch.setattr(module, "_PyannotePipeline", DummyPipeline, raising=False)
    monkeypatch.setattr(module, "_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(module, "torch", None, raising=False)
    monkeypatch.setattr(module, "Annotation", DummyAnnotation, raising=False)


def test_build_pyannote_diarizer_runs_with_mock(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF0000")

    config = {
        "providers": {
            "pyannote": {
                "enabled": True,
                "model": "pyannote/speaker-diarization@2.1",
                "auth_token_env": "SHOW_SCRIBE_PYANNOTE_TOKEN",
                "segmentation_threshold": 0.42,
            },
        },
        "diarization": {
            "min_speakers": 1,
            "max_speakers": 3,
            "onset": 0.25,
            "offset": 0.15,
            "overlap_threshold": 0.55,
        },
        "runtime": {
            "prefer_gpu": False,
            "device_priority": ("cpu",),
        },
    }

    diarizer: PyannoteDiarizer = build_pyannote_diarizer(config)
    result = diarizer.diarize(audio_path)

    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_A"
    assert result.metadata.speaker_count == 2
    assert result.metadata.parameters["device"] == "cpu"
    assert result.metadata.parameters["segmentation_threshold"] == 0.42
    assert result.metadata.parameters["onset"] == 0.25
    assert result.metadata.parameters["overlap_threshold"] == 0.55


def test_missing_token_raises_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("SHOW_SCRIBE_PYANNOTE_TOKEN", raising=False)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")

    config = {
        "providers": {
            "pyannote": {
                "enabled": True,
                "model": "pyannote/speaker-diarization@2.1",
                "auth_token_env": "SHOW_SCRIBE_PYANNOTE_TOKEN",
            },
        },
        "diarization": {},
        "runtime": {},
    }

    diarizer = build_pyannote_diarizer(config)
    with pytest.raises(PyannoteNotAvailableError):
        diarizer.diarize(audio_path)
