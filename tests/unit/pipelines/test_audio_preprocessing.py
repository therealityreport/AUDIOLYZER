"""Unit tests for the audio preprocessing pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from show_scribe.pipelines.audio_preprocessing import AudioPreprocessor, PreprocessingArtifacts


def _make_wave(path: Path, duration: float = 1.0, sample_rate: int = 16_000) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, waveform.astype(np.float32), sample_rate)


def test_preprocess_creates_expected_artifacts(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    _make_wave(source)

    config = {
        "audio_preprocessing": {
            "enable": True,
            "vocal_separation": {"enable": True},
            "enhancement": {"enable": True},
        }
    }
    preprocessor = AudioPreprocessor(config)

    def fake_analysis(self, path: str) -> dict[str, float]:
        return {
            "snr": 10.0,
            "music_ratio": 0.8,
            "speech_clarity": 0.4,
            "reverb_score": 0.6,
            "duration_analyzed": 1.0,
        }

    def copy_file(_self, audio_path: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, destination)
        return destination

    monkeypatch.setattr(AudioPreprocessor, "analyze_audio_quality", fake_analysis)
    monkeypatch.setattr(AudioPreprocessor, "separate_vocals", copy_file)
    monkeypatch.setattr(AudioPreprocessor, "enhance_audio", copy_file)

    output_dir = tmp_path / "processed"
    artifacts = preprocessor.preprocess(str(source), output_dir=str(output_dir))
    assert isinstance(artifacts, PreprocessingArtifacts)

    expected_processed = output_dir / "audio_processed.wav"
    expected_vocals = output_dir / "audio_vocals.wav"
    expected_enhanced = output_dir / "audio_enhanced_vocals.wav"

    assert artifacts.final_audio == expected_processed
    assert artifacts.vocals_audio == expected_vocals
    assert artifacts.enhanced_audio == expected_enhanced
    assert artifacts.enhanced_mix_audio is None

    assert expected_processed.exists()
    assert expected_vocals.exists()
    assert expected_enhanced.exists()
    assert artifacts.report_path.exists()

    assert artifacts.report["steps_applied"] == ["vocal_separation", "enhancement"]
    assert artifacts.report["preprocessing_enabled"] is True
    assert artifacts.report["files"]["enhanced_mix"] is None
    cleanup = artifacts.report["cleanup"]
    assert cleanup["intermediates_retained"] is True
    assert cleanup["intermediates_purged"] is False


def test_preprocess_disabled_creates_report(tmp_path) -> None:
    source = tmp_path / "input.wav"
    _make_wave(source)

    config = {
        "audio_preprocessing": {
            "enable": False,
        }
    }
    preprocessor = AudioPreprocessor(config)

    output_dir = tmp_path / "processed"
    artifacts = preprocessor.preprocess(str(source), output_dir=str(output_dir))

    assert artifacts.vocals_audio is None
    assert artifacts.enhanced_audio is None
    assert artifacts.enhanced_mix_audio is None
    assert artifacts.enhanced_mix_audio is None

    assert artifacts.final_audio.exists()
    assert artifacts.report_path.exists()

    with artifacts.report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["preprocessing_enabled"] is False
    assert payload["steps_applied"] == []
    assert payload["files"]["processed"] == str(artifacts.final_audio)
    assert payload["files"].get("enhanced_mix") is None
    cleanup = payload["cleanup"]
    assert cleanup["intermediates_retained"] is True
    assert cleanup["intermediates_purged"] is False


def test_preprocess_purges_intermediates_when_configured(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    _make_wave(source)

    config = {
        "audio_preprocessing": {
            "enable": True,
            "retain_intermediates": False,
            "vocal_separation": {"enable": True},
            "enhancement": {"enable": True},
        }
    }
    preprocessor = AudioPreprocessor(config)

    def fake_analysis(self, path: str) -> dict[str, float]:
        return {
            "snr": 8.0,
            "music_ratio": 0.9,
            "speech_clarity": 0.3,
            "reverb_score": 0.7,
            "duration_analyzed": 1.0,
        }

    def copy_file(_self, audio_path: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, destination)
        return destination

    monkeypatch.setattr(AudioPreprocessor, "analyze_audio_quality", fake_analysis)
    monkeypatch.setattr(AudioPreprocessor, "separate_vocals", copy_file)
    monkeypatch.setattr(AudioPreprocessor, "enhance_audio", copy_file)

    output_dir = tmp_path / "processed"
    artifacts = preprocessor.preprocess(str(source), output_dir=str(output_dir))

    assert artifacts.vocals_audio is None
    assert artifacts.enhanced_audio is None
    assert not (output_dir / "audio_vocals.wav").exists()
    assert not (output_dir / "audio_enhanced_vocals.wav").exists()
    assert not (output_dir / "audio_enhanced_mix.wav").exists()

    cleanup = artifacts.report["cleanup"]
    assert cleanup["intermediates_retained"] is False
    assert cleanup["intermediates_purged"] is True


def test_preprocess_uses_clearervoice_wrapper(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    _make_wave(source)

    config = {
        "audio_preprocessing": {
            "enable": True,
            "retain_intermediates": True,
            "vocal_separation": {"enable": True},
            "enhancement": {"enable": True, "provider": "clearervoice"},
        }
    }
    preprocessor = AudioPreprocessor(config)

    def fake_analysis(_self, _path: str) -> dict[str, float]:
        return {
            "snr": 5.0,
            "music_ratio": 0.8,
            "speech_clarity": 0.3,
            "reverb_score": 0.8,
            "duration_analyzed": 1.0,
        }

    processed_waveform = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16_000, endpoint=False))

    class StubWrapper:
        def __init__(self) -> None:
            self.sep_calls = 0
            self.vocals_calls = 0
            self.mix_calls = 0

        def separate_vocals(self, audio_path: Path, destination: Path) -> Path:
            self.sep_calls += 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, processed_waveform, 16_000)
            return destination

        def enhance_vocals(self, audio_path: Path, destination: Path) -> Path:
            self.vocals_calls += 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, processed_waveform * 0.5, 16_000)
            return destination

        def enhance_mix(self, audio_path: Path, destination: Path) -> Path:
            self.mix_calls += 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, processed_waveform * 0.75, 16_000)
            return destination

    stub_wrapper = StubWrapper()

    def _unexpected_enhance(*_args, **_kwargs):
        raise AssertionError("Resemble enhancer should not be invoked for ClearerVoice provider")

    monkeypatch.setattr(AudioPreprocessor, "analyze_audio_quality", fake_analysis)
    monkeypatch.setattr(AudioPreprocessor, "enhance_audio", _unexpected_enhance)
    monkeypatch.setattr(AudioPreprocessor, "_get_clearervoice_wrapper", lambda self: stub_wrapper)

    output_dir = tmp_path / "processed"
    artifacts = preprocessor.preprocess(str(source), output_dir=str(output_dir))

    assert stub_wrapper.sep_calls == 1
    assert stub_wrapper.vocals_calls == 1
    assert stub_wrapper.mix_calls == 1

    enhanced_vocals = output_dir / "audio_enhanced_vocals.wav"
    enhanced_mix = output_dir / "audio_enhanced_mix.wav"

    assert artifacts.enhanced_audio == enhanced_vocals
    assert artifacts.enhanced_mix_audio == enhanced_mix
    assert enhanced_vocals.exists()
    assert enhanced_mix.exists()
    assert artifacts.report["files"]["enhanced_mix"] == str(enhanced_mix)
