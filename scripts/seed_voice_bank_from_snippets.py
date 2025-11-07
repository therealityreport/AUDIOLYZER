#!/usr/bin/env python
"""Seed the voice bank using pre-generated snippet audio files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from show_scribe.config.load import load_config
from show_scribe.pipelines.speaker_id.voice_bank import VoiceBankPipeline
from show_scribe.storage.db import SQLiteDatabase
from show_scribe.storage.paths import build_paths
from show_scribe.storage.voice_bank_manager import VoiceBankManager, normalize_key
from show_scribe.utils.audio_io import load_audio


def _parse_assignments(entries: Iterable[str]) -> list[tuple[str, Path]]:
    assignments: list[tuple[str, Path]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid assignment '{entry}'. Use SPEAKER=path/to/snippet.wav")
        speaker, path = entry.split("=", 1)
        speaker = speaker.strip()
        path = path.strip()
        if not speaker or not path:
            raise ValueError(f"Invalid assignment '{entry}'. Entries cannot be empty.")
        assignments.append((speaker, Path(path)))
    if not assignments:
        raise ValueError("At least one --sample entry is required.")
    return assignments


def _load_show_cast(show_config_path: Path) -> dict[str, dict[str, object]]:
    with show_config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    cast_lookup: dict[str, dict[str, object]] = {}
    for entry in payload.get("cast_members", []) or []:
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("canonical_name")
        if canonical:
            cast_lookup[str(canonical).casefold()] = entry
    return cast_lookup


def _ensure_embedding_capacity(
    manager: VoiceBankManager, speaker_id: int, max_embeddings: int
) -> None:
    entries = manager.list_embeddings(speaker_id)
    if len(entries) <= max_embeddings:
        return
    for entry in entries[max_embeddings:]:
        if entry.id is not None:
            manager.remove_embedding(entry.id)
        try:
            entry.embedding_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", default="dev", help="Configuration environment to load (default: dev)."
    )
    parser.add_argument(
        "--episode-id",
        required=True,
        help="Episode identifier used for naming stored embeddings.",
    )
    parser.add_argument(
        "--show-config",
        type=Path,
        help="Optional explicit path to show_config.json (defaults to inferred location).",
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=[],
        metavar="SPEAKER=snippet.wav",
        help="Assign a snippet WAV to a speaker (may be specified multiple times).",
    )
    args = parser.parse_args()

    if not args.samples:
        parser.error("Provide at least one --sample SPEAKER=path/to/snippet.wav entry.")

    assignments = _parse_assignments(args.samples)

    config = load_config(args.env)
    paths = build_paths(config)

    show_config_path = args.show_config
    if show_config_path is None:
        show_root = paths.data_root / "shows"
        inferred = None
        for candidate in show_root.glob("*/show_config.json"):
            inferred = candidate
            break
        if inferred is None:
            parser.error("Unable to infer show_config.json; pass --show-config explicitly.")
        show_config_path = inferred

    show_config_path = show_config_path.expanduser().resolve()
    cast_lookup = _load_show_cast(show_config_path)

    database = SQLiteDatabase(paths.voice_bank_db)
    if not database.db_path.exists():
        database.initialize()
    manager = VoiceBankManager(database)

    pipeline = VoiceBankPipeline(config, paths, voice_bank=manager)
    encoder = pipeline.embedding_backend
    embedding_root = pipeline.embedding_root
    embedding_root.mkdir(parents=True, exist_ok=True)

    max_embeddings = pipeline.settings.max_embeddings_per_speaker

    for speaker_name, snippet_path in assignments:
        snippet_path = snippet_path.expanduser().resolve()
        if not snippet_path.exists():
            raise FileNotFoundError(f"Snippet not found: {snippet_path}")

        clip = load_audio(
            snippet_path,
            mono=True,
            target_sample_rate=encoder.target_sample_rate,
        )
        if clip.samples.size == 0:
            raise ValueError(f"Snippet has no audio samples: {snippet_path}")

        embedding = encoder.encode(clip)

        key = normalize_key(speaker_name)
        cast_entry = cast_lookup.get(speaker_name.casefold(), {})
        aliases = cast_entry.get("aliases", []) if isinstance(cast_entry, dict) else []
        misspellings = (
            cast_entry.get("common_misspellings", []) if isinstance(cast_entry, dict) else []
        )

        profile = manager.upsert_speaker(
            display_name=speaker_name,
            key=key,
            aliases=aliases,
            misspellings=misspellings,
        )

        if profile.id is None:
            raise RuntimeError(f"Failed to persist speaker profile for {speaker_name}.")

        speaker_dir = embedding_root / key
        speaker_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = speaker_dir / f"{args.episode_id}_{snippet_path.stem}.npy"
        np.save(embedding_path, embedding.astype(np.float32, copy=False))

        manager.add_embedding(
            speaker_id=profile.id,
            embedding_path=embedding_path,
            source_episode=args.episode_id,
            source_timestamp=None,
        )
        manager.increment_segment_count(profile.key)
        _ensure_embedding_capacity(manager, profile.id, max_embeddings)
        print(f"[INFO] Seeded embedding for {speaker_name} from {snippet_path} -> {embedding_path}")

    print("[INFO] Voice bank updated. Re-run the pipeline to apply new speaker assignments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
