from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest

from pipeline import audio_ops


def _write_silence_wav(path: Path, *, duration_seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(16000 * duration_seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


@pytest.fixture
def media_fixture() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "tests" / "fixtures" / "video" / "test_clip.mp4"


@pytest.fixture
def preset_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "reality_tv.yaml"


def test_extract_and_enhance_produces_artifacts(
    tmp_path: Path, media_fixture: Path, preset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_dir = tmp_path / "episode"

    monkeypatch.setattr(audio_ops, "which", lambda _: "/usr/bin/ffmpeg")

    def _fake_run(cmd: list[str], check: bool = True, **_: object):
        if cmd and cmd[0] == "ffmpeg":
            target = Path(cmd[-1])
            _write_silence_wav(target)
        elif "-m" in cmd:
            module = cmd[cmd.index("-m") + 1]
            output_arg = cmd.index("--output") + 1
            target = Path(cmd[output_arg])
            _write_silence_wav(target)
            if module == "preproc.enhance":
                vocals_arg = cmd.index("--input") + 1
                vocals_source = Path(cmd[vocals_arg])
                if not vocals_source.exists():
                    _write_silence_wav(vocals_source)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(audio_ops.subprocess, "run", _fake_run)

    result = audio_ops.extract_and_enhance(media_fixture, outputs_dir, preset_path)

    extracted = outputs_dir / "audio_extracted.wav"
    vocals = outputs_dir / "audio_vocals.wav"
    enhanced = outputs_dir / "audio_enhanced_vocals.wav"
    enhanced_mix = outputs_dir / "audio_enhanced_mix.wav"
    report = outputs_dir / "preprocessing_report.json"

    assert extracted.exists()
    assert vocals.exists()
    assert enhanced.exists()
    assert enhanced_mix.exists()
    assert report.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["outputs"]["vocals"] == str(vocals)
    assert data["outputs"]["enhanced_vocals"] == str(enhanced)
    assert data["outputs"]["enhanced_mix"] == str(enhanced_mix)
    assert data["preset"] == str(preset_path)

    assert result["extracted"] == extracted
    assert result["vocals"] == vocals
    assert result["enhanced_vocals"] == enhanced
    assert result["enhanced_mix"] == enhanced_mix
    assert result["report"] == report
