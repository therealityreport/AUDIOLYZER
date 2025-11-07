"""Integration test for the audio extraction pipeline using FFmpeg."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from show_scribe.pipelines.extract_audio import build_extractor
from show_scribe.storage.naming import EpisodeDescriptor
from show_scribe.storage.paths import PathsConfig


@pytest.mark.integration
def test_extract_audio_pipeline(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg binaries not available.")

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
            "min_peak_dbfs": -60.0,
            "min_rms": 1e-6,
            "enforce_strict": False,
        },
    }

    extractor = build_extractor(config, paths)

    source_video = Path(__file__).parents[1] / "fixtures" / "video" / "sample_pattern.mp4"
    descriptor = EpisodeDescriptor(show_name="Sample Show", season=1, episode=1)

    progress_updates: list[float] = []

    def progress_callback(update):
        if update.out_time is not None:
            progress_updates.append(update.out_time)

    result = extractor.extract(source_video, descriptor, progress=progress_callback)

    assert result.audio_path.exists()
    assert result.quality_report.checks, "Expected quality checks."
    assert progress_updates, "Expected at least one progress update."
