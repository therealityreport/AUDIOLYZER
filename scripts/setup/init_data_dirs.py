#!/usr/bin/env python3
"""Initialize the Show-Scribe data directory structure."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from show_scribe.config.load import load_config
from show_scribe.storage.paths import PathsConfig, build_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the standard directory structure used by Show-Scribe."
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Configuration environment to load (default: %(default)s).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Optional configuration directory override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the directories that would be created without writing to disk.",
    )
    args = parser.parse_args()

    config = load_config(args.env, config_dir=args.config_dir)
    paths = build_paths(config)

    if args.dry_run:
        print("Dry run: no directories will be created.")

    ensure_base_directories(paths, dry_run=args.dry_run)
    ensure_data_structure(paths, dry_run=args.dry_run)
    if not args.dry_run:
        print("✔ Data directory initialization complete.")


def ensure_base_directories(paths: PathsConfig, *, dry_run: bool) -> None:
    """Ensure top-level directories defined by configuration exist."""
    if dry_run:
        for directory in _base_directories(paths):
            print(f"[dry-run] would create {directory}")
        return

    paths.ensure_directories()


def ensure_data_structure(paths: PathsConfig, *, dry_run: bool) -> None:
    """Create nested data directories used by voice bank and episode storage."""
    directories: list[Path] = [
        paths.data_root / "shows",
        paths.data_root / "samples",
        paths.data_root / "voice_bank",
        paths.voice_bank_backup_dir(),
        paths.data_root / "voice_bank" / "embeddings",
        paths.data_root / "voice_bank" / "audio_samples",
        paths.data_root / "voice_bank" / "profiles",
        paths.cache_dir / "waveforms",
        paths.cache_dir / "spectrograms",
        paths.temp_dir / "ffmpeg",
        paths.temp_dir / "work",
    ]

    for directory in directories:
        if dry_run:
            print(f"[dry-run] would create {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)


def _base_directories(paths: PathsConfig) -> Iterable[Path]:
    """Yield the foundational directories managed by PathsConfig."""
    yield paths.data_root
    yield paths.output_root
    yield paths.cache_dir
    yield paths.temp_dir
    yield paths.models_dir
    yield paths.voice_bank_db.parent
    yield paths.logs_dir


if __name__ == "__main__":
    main()
