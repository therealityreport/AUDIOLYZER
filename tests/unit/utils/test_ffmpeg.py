"""Tests for the FFmpeg helper utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from show_scribe.utils.ffmpeg import FFmpeg, FFmpegError, FFmpegProgress


class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeProcess:
    """Minimal stub that mimics subprocess.Popen for progress parsing."""

    def __init__(self, lines: list[str]) -> None:
        self._lines: Iterator[str] = iter(lines)
        self.stdout = self
        self.stderr_content = ""
        self.returncode = 0
        self._closed = False

    def readline(self) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            self._closed = True
            return ""

    def poll(self) -> int | None:
        return 0 if self._closed else None

    def communicate(self) -> tuple[str, str]:
        return "", self.stderr_content

    def kill(self) -> None:
        self.returncode = -9
        self._closed = True


def test_probe_returns_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata = {"streams": [], "format": {}}

    def fake_run(*_args, **_kwargs):
        return FakeCompletedProcess(returncode=0, stdout=json.dumps(metadata))

    monkeypatch.setattr("subprocess.run", fake_run)
    probe = FFmpeg().probe(tmp_path / "video.mp4")
    assert probe == metadata


def test_probe_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*_args, **_kwargs):
        return FakeCompletedProcess(returncode=1, stderr="not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(FFmpegError):
        FFmpeg().probe(tmp_path / "video.mp4")


def test_extract_audio_invokes_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(self, args, progress_callback=None, timeout=None):
        captured["args"] = list(args)
        captured["progress_callback"] = progress_callback
        captured["timeout"] = timeout

    monkeypatch.setattr(FFmpeg, "run", fake_run)
    ffmpeg = FFmpeg()
    ffmpeg.extract_audio(tmp_path / "input.mp4", tmp_path / "output.wav")

    assert captured["args"][0] == "-i"
    assert tmp_path / "output.wav" == Path(captured["args"][-1])
    assert captured["progress_callback"] is None


def test_run_with_progress_parses_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[FFmpegProgress] = []

    def fake_popen(*_args, **_kwargs):
        lines = [
            "frame=1\n",
            "fps=50.0\n",
            "out_time=00:00:01.00\n",
            "progress=continue\n",
            "frame=2\n",
            "fps=75.0\n",
            "out_time=00:00:02.00\n",
            "progress=end\n",
        ]
        return FakeProcess(lines)

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    ffmpeg = FFmpeg()
    ffmpeg.run(
        ["-i", "input", "output"],
        progress_callback=lambda progress: updates.append(progress),
    )

    assert updates, "Expected progress updates."
    assert updates[-1].status == "end"
    assert updates[-1].out_time == pytest.approx(2.0, abs=1e-2)
