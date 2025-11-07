"""Tests for file naming helpers."""

from __future__ import annotations

from pathlib import Path

from show_scribe.storage.naming import (
    EpisodeDescriptor,
    build_artifact_filename,
    build_episode_id,
    resolve_artifact_path,
)
from show_scribe.storage.paths import PathsConfig


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


def test_build_episode_id() -> None:
    descriptor = EpisodeDescriptor(show_name="The Office", season=2, episode=5)
    assert build_episode_id(descriptor) == "TheOffice_S02E05"


def test_build_artifact_filename_with_variant() -> None:
    descriptor = EpisodeDescriptor(show_name="The Office", season=2, episode=5, variant="extended")
    filename = build_artifact_filename(descriptor, "audio_extracted", "wav")
    assert filename == "TheOffice_S02E05_extended_audio_extracted.wav"


def test_resolve_artifact_path_creates_directory(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    descriptor = EpisodeDescriptor(show_name="The Office", season=1, episode=1)
    artifact_path = resolve_artifact_path(
        paths,
        descriptor,
        "audio_extracted",
        "wav",
        ensure_directory=True,
    )
    assert artifact_path.parent.exists()
    assert artifact_path.name == "TheOffice_S01E01_audio_extracted.wav"
