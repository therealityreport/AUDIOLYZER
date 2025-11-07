"""Unit tests for the audio extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from show_scribe.pipelines.audio_preprocessing import PreprocessingArtifacts
from show_scribe.pipelines.extract_audio import (
    AudioExtractionConfig,
    AudioExtractionError,
    AudioExtractor,
    QualityThresholds,
)
from show_scribe.storage.naming import EpisodeDescriptor
from show_scribe.storage.paths import PathsConfig
from show_scribe.utils.ffmpeg import FFmpegError


class StubFFmpeg:
    def __init__(self, waveform: np.ndarray) -> None:
        self.waveform = waveform
        self.calls = 0

    def extract_audio(self, input_path, output_path, **kwargs):
        self.calls += 1
        sample_rate = kwargs.get("sample_rate", 16_000)
        sf.write(output_path, self.waveform, sample_rate)


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


def make_config(
    enforce_strict: bool = False, enable_preprocessing: bool = False
) -> AudioExtractionConfig:
    config = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "codec": "pcm_s16le",
            "normalization": {
                "target_lufs": -20.0,
                "loudness_range": 7.0,
                "true_peak": -1.0,
            },
        },
        "audio_quality": {
            "min_duration_seconds": 0.5,
            "min_peak_dbfs": -30.0,
            "min_rms": 1e-5,
            "enforce_strict": enforce_strict,
        },
        "audio_preprocessing": {"enable": enable_preprocessing},
    }
    return AudioExtractionConfig.from_config(config)


def make_source(tmp_path: Path, suffix: str = ".mp4") -> Path:
    path = tmp_path / f"input{suffix}"
    path.write_bytes(b"fake")
    return path


def test_extract_audio_success(tmp_path: Path) -> None:
    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.1 * np.sin(2 * np.pi * 440 * t)

    ffmpeg = StubFFmpeg(waveform.astype(np.float32))
    extractor = AudioExtractor(make_paths(tmp_path), make_config(), ffmpeg=ffmpeg)
    descriptor = EpisodeDescriptor(show_name="The Office", season=1, episode=1)

    result = extractor.extract(make_source(tmp_path), descriptor)

    assert result.audio_path.exists()
    assert result.clip.sample_rate == sr
    assert result.quality_report.passed
    assert ffmpeg.calls == 1


def test_extract_audio_enforces_quality(tmp_path: Path) -> None:
    waveform = np.zeros(16000, dtype=np.float32)
    config = make_config(enforce_strict=True)
    thresholds = QualityThresholds(
        min_duration_seconds=0.5,
        min_peak_dbfs=-30.0,
        min_rms=0.01,
        enforce_strict=True,
    )
    config.thresholds = thresholds

    extractor = AudioExtractor(make_paths(tmp_path), config, ffmpeg=StubFFmpeg(waveform))
    descriptor = EpisodeDescriptor(show_name="The Office", season=1, episode=1)

    with pytest.raises(AudioExtractionError):
        extractor.extract(make_source(tmp_path), descriptor)


def test_unsupported_extension(tmp_path: Path) -> None:
    extractor = AudioExtractor(
        make_paths(tmp_path),
        make_config(),
        ffmpeg=StubFFmpeg(np.zeros(1000, dtype=np.float32)),
    )
    descriptor = EpisodeDescriptor(show_name="The Office", season=1, episode=1)

    with pytest.raises(AudioExtractionError):
        extractor.extract(make_source(tmp_path, suffix=".txt"), descriptor)


def test_ffmpeg_failure(tmp_path: Path) -> None:
    class FailingFFmpeg(StubFFmpeg):
        def extract_audio(self, *args, **kwargs):
            raise FFmpegError("boom")

    extractor = AudioExtractor(
        make_paths(tmp_path),
        make_config(),
        ffmpeg=FailingFFmpeg(np.zeros(1, dtype=np.float32)),
    )
    descriptor = EpisodeDescriptor(show_name="The Office", season=1, episode=1)

    with pytest.raises(AudioExtractionError):
        extractor.extract(make_source(tmp_path), descriptor)


def test_extract_audio_uses_preprocessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    processed_waveform = waveform * 0.5

    class DummyPreprocessor:
        def __init__(self, config):
            self.config = config

        def preprocess(self, audio_path: str, output_dir: str) -> PreprocessingArtifacts:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            vocals = output_dir_path / "audio_vocals.wav"
            enhanced = output_dir_path / "audio_enhanced_vocals.wav"
            enhanced_mix = output_dir_path / "audio_enhanced_mix.wav"
            processed = output_dir_path / "audio_processed.wav"
            for target in (vocals, enhanced, enhanced_mix, processed):
                sf.write(target, processed_waveform, sr)

            report_path = output_dir_path / "preprocessing_report.json"
            report_payload = {
                "steps_applied": ["vocal_separation", "enhancement"],
                "preprocessing_enabled": True,
                "timings_seconds": {"vocal_separation": 0.1, "enhancement": 0.2},
                "files": {
                    "processed": str(processed),
                    "vocals": str(vocals),
                    "enhanced": str(enhanced),
                    "enhanced_mix": str(enhanced_mix),
                    "report": str(report_path),
                },
                "cleanup": {
                    "intermediates_retained": True,
                    "intermediates_purged": False,
                },
            }
            report_path.write_text(json.dumps(report_payload), encoding="utf-8")
            return PreprocessingArtifacts(
                final_audio=processed,
                vocals_audio=vocals,
                enhanced_audio=enhanced,
                enhanced_mix_audio=enhanced_mix,
                report_path=report_path,
                report=report_payload,
            )

    monkeypatch.setattr(
        "show_scribe.pipelines.audio_preprocessing.AudioPreprocessor",
        DummyPreprocessor,
    )

    extractor = AudioExtractor(
        make_paths(tmp_path),
        make_config(enable_preprocessing=True),
        ffmpeg=StubFFmpeg(waveform),
        full_config={"audio_preprocessing": {"enable": True}},
    )
    descriptor = EpisodeDescriptor(show_name="Benchmark", season=1, episode=1)

    result = extractor.extract(make_source(tmp_path), descriptor)

    assert result.audio_path.name.endswith("audio_processed.wav")
    assert result.preprocessing_report is not None
    assert result.preprocessing_report_path is not None

    files = result.preprocessing_report.get("files", {})
    assert files.get("episode_artifact") == str(result.audio_path)
    assert Path(files.get("vocals", "")).exists()
    assert Path(files.get("enhanced", "")).exists()
    assert Path(files.get("enhanced_mix", "")).exists()
    assert Path(result.preprocessing_report_path).exists()

    processed_audio, _ = sf.read(result.audio_path)
    assert np.isclose(processed_audio.max(), processed_waveform.max(), atol=1e-4)

    episode_dir = extractor.paths.episode_directory(descriptor.show_name, descriptor.episode_id)
    processed_dir = episode_dir / "processed_audio"
    assert (processed_dir / "audio_vocals.wav").exists()
    assert (processed_dir / "audio_enhanced_vocals.wav").exists()
    assert (processed_dir / "audio_enhanced_mix.wav").exists()
