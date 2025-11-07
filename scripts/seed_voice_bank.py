#!/usr/bin/env python
"""Seed the voice bank with embeddings derived from an aligned transcript."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from show_scribe.config.load import load_config
from show_scribe.pipelines.orchestrator import _infer_show_config_path
from show_scribe.pipelines.speaker_id.voice_bank import VoiceBankPipeline
from show_scribe.storage.db import SQLiteDatabase
from show_scribe.storage.paths import build_paths
from show_scribe.storage.voice_bank_manager import VoiceBankManager, normalize_key
from show_scribe.utils.audio_io import AudioClip, extract_segment, load_audio


def _parse_mapping(entries: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid mapping entry '{entry}'. Expected form CLUSTER=Name.")
        cluster, name = entry.split("=", 1)
        cluster = cluster.strip()
        name = name.strip()
        if not cluster or not name:
            raise ValueError(f"Invalid mapping entry '{entry}'. Entries cannot be empty.")
        mapping[cluster] = name
    if not mapping:
        raise ValueError("At least one --map entry is required.")
    return mapping


def _load_show_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Show config at {path} must be a JSON object.")
    return data


def _build_cast_lookup(show_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    cast_members = show_config.get("cast_members") or []
    for entry in cast_members:
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("canonical_name")
        if not canonical:
            continue
        lookup[str(canonical).casefold()] = entry
    return lookup


def _gather_segments(transcript_path: Path) -> dict[str, list[tuple[float, float]]]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        speaker = segment.get("speaker")
        metadata = segment.get("metadata") or {}
        original = metadata.get("original_speaker") or speaker
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(original, str) or start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        grouped[original].append((start_f, end_f))
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", default="dev", help="Configuration environment to load (default: dev)."
    )
    parser.add_argument(
        "--episode-id",
        required=True,
        help="Episode identifier (e.g., RHOBH_S05E01_fullscene).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to the extracted episode audio (defaults to episode directory).",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Path to the transcript_final.json file (defaults to episode directory).",
    )
    parser.add_argument(
        "--show-config",
        type=Path,
        help="Optional explicit path to show_config.json.",
    )
    parser.add_argument(
        "--map",
        dest="mappings",
        action="append",
        default=[],
        metavar="CLUSTER=Name",
        help="Map a diarization cluster (e.g., SPEAKER_01) to a canonical speaker name.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=8,
        help="Maximum number of transcript segments per speaker to use when generating the embedding.",
    )
    args = parser.parse_args()

    if not args.mappings:
        parser.error("At least one --map entry is required to seed the voice bank.")

    mapping = _parse_mapping(args.mappings)

    config = load_config(args.env)
    paths = build_paths(config)

    show_config_path = args.show_config
    if show_config_path is None:
        inferred = _infer_show_config_path(args.episode_id, paths)
        if inferred is None:
            parser.error("Unable to infer show_config.json; provide --show-config explicitly.")
        show_config_path = inferred

    show_config = _load_show_config(show_config_path)
    cast_lookup = _build_cast_lookup(show_config)
    show_name = show_config.get("show_slug") or show_config.get("show_name")
    if not show_name:
        parser.error("Show config missing 'show_slug' or 'show_name'.")

    episode_dir = paths.episode_directory(str(show_name), args.episode_id)
    default_audio = episode_dir / f"{args.episode_id}_audio_extracted.wav"
    audio_path = args.audio or default_audio
    transcript_path = args.transcript or (episode_dir / "transcript_final.json")

    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")
    if not transcript_path.exists():
        parser.error(f"Transcript JSON not found: {transcript_path}")

    database = SQLiteDatabase(paths.voice_bank_db)
    if not database.db_path.exists():
        database.initialize()
    manager = VoiceBankManager(database)

    pipeline = VoiceBankPipeline(config, paths, voice_bank=manager)
    encoder = pipeline.embedding_backend
    clip = load_audio(audio_path, target_sample_rate=encoder.target_sample_rate)
    segment_lookup = _gather_segments(transcript_path)

    max_embeddings = pipeline.settings.max_embeddings_per_speaker

    embedding_root = pipeline.embedding_root
    embedding_root.mkdir(parents=True, exist_ok=True)

    def _ensure_capacity(profile_id: int) -> None:
        entries = manager.list_embeddings(profile_id)
        if len(entries) <= max_embeddings:
            return
        for entry in entries[max_embeddings:]:
            if entry.id is not None:
                manager.remove_embedding(entry.id)
            try:
                entry.embedding_path.unlink(missing_ok=True)
            except Exception:
                pass

    seeded = 0
    for cluster_id, speaker_name in mapping.items():
        segments = segment_lookup.get(cluster_id)
        if not segments:
            print(f"[WARN] No segments found for cluster {cluster_id}; skipping {speaker_name}.")
            continue

        segments = sorted(segments, key=lambda item: item[0])
        if args.max_segments and len(segments) > args.max_segments:
            segments = segments[: args.max_segments]

        waveforms: list[np.ndarray] = []
        for start, end in segments:
            extract = extract_segment(clip, start, end)
            if extract.samples.size == 0:
                continue
            waveform = extract.samples
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0, dtype=np.float32)
            waveforms.append(waveform.astype(np.float32, copy=False))

        if not waveforms:
            print(f"[WARN] Unable to extract audio for {speaker_name}; skipping.")
            continue

        concatenated = np.concatenate(waveforms, axis=-1)
        embedding_clip = AudioClip(samples=concatenated, sample_rate=clip.sample_rate)
        embedding = encoder.encode(embedding_clip)

        canonical_key = normalize_key(speaker_name)
        cast_entry = cast_lookup.get(speaker_name.casefold())
        aliases = cast_entry.get("aliases", []) if cast_entry else []
        misspellings = cast_entry.get("common_misspellings", []) if cast_entry else []

        profile = manager.upsert_speaker(
            display_name=speaker_name,
            key=canonical_key,
            aliases=aliases,
            misspellings=misspellings,
        )
        if profile.id is None:
            print(f"[WARN] Unable to persist profile for {speaker_name}; skipping.")
            continue

        speaker_dir = embedding_root / canonical_key
        speaker_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = speaker_dir / f"{args.episode_id}_{cluster_id}.npy"
        np.save(embedding_path, embedding.astype(np.float32, copy=False))

        manager.add_embedding(
            speaker_id=profile.id,
            embedding_path=embedding_path,
            source_episode=args.episode_id,
        )
        manager.increment_segment_count(profile.key, increment=len(segments))
        _ensure_capacity(profile.id)
        seeded += 1
        print(f"[INFO] Seeded voice bank entry for {speaker_name} ({cluster_id}).")

    if seeded == 0:
        print("[WARN] No embeddings were seeded. Verify the mapping and transcript inputs.")
    else:
        print(f"[INFO] Seeded embeddings for {seeded} speaker(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
