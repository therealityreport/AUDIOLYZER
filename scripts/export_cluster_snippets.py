#!/usr/bin/env python
"""Export representative audio snippets for each diarization cluster."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from show_scribe.config.load import load_config
from show_scribe.pipelines.orchestrator import _infer_show_config_path
from show_scribe.storage.paths import build_paths
from show_scribe.utils.audio_io import AudioClip, extract_segment, load_audio, save_audio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dev", help="Configuration environment (default: dev).")
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
        help="Path to transcript_final.json (defaults to episode directory).",
    )
    parser.add_argument(
        "--show-config",
        type=Path,
        help="Optional explicit path to show_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/speaker_snippets"),
        help="Directory for exported snippets (default: exports/speaker_snippets).",
    )
    parser.add_argument(
        "--snippets-per-cluster",
        type=int,
        default=5,
        help="Maximum number of segments per cluster to export (default: 5).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Skip segments shorter than this duration in seconds (default: 1.5).",
    )
    return parser.parse_args()


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Mapping[str, object]]:
    config = load_config(args.env)
    paths = build_paths(config)

    show_config_path = args.show_config
    if show_config_path is None:
        inferred = _infer_show_config_path(args.episode_id, paths)
        if inferred is None:
            raise SystemExit("Unable to infer show_config.json; provide --show-config explicitly.")
        show_config_path = inferred

    show_config_path = show_config_path.expanduser().resolve()
    if not show_config_path.exists():
        raise SystemExit(f"Show config not found: {show_config_path}")

    with show_config_path.open("r", encoding="utf-8") as handle:
        show_config = json.load(handle) or {}

    show_slug = show_config.get("show_slug") or show_config.get("show_name")
    if not show_slug:
        raise SystemExit("Show config missing 'show_slug' or 'show_name'.")

    episode_dir = paths.episode_directory(str(show_slug), args.episode_id)
    audio_path = args.audio or (episode_dir / f"{args.episode_id}_audio_extracted.wav")
    transcript_path = args.transcript or (episode_dir / "transcript_final.json")
    return audio_path, transcript_path, show_config


def _load_transcript_segments(transcript_path: Path) -> list[dict[str, object]]:
    with transcript_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    segments = payload.get("segments") or []
    result: list[dict[str, object]] = []
    for segment in segments:
        if isinstance(segment, dict):
            result.append(segment)
    return result


def _cluster_key(segment: Mapping[str, object]) -> str:
    metadata = segment.get("metadata") or {}
    speaker = metadata.get("original_speaker")
    if isinstance(speaker, str) and speaker.strip():
        return speaker.strip()
    speaker = segment.get("speaker")
    if isinstance(speaker, str) and speaker.strip():
        return speaker.strip()
    return "unknown"


def _resolve_primary_speaker(segment: Mapping[str, object]) -> str:
    metadata = segment.get("metadata") or {}
    speaker = metadata.get("original_speaker") or segment.get("speaker")
    if isinstance(speaker, str) and speaker.strip():
        return speaker.strip()
    return "unknown"


def _trim_segment(
    segment: Mapping[str, object],
    start: float,
    end: float,
) -> tuple[float, bool]:
    metadata = segment.get("metadata") or {}
    distribution = metadata.get("speaker_distribution") or {}
    primary = _resolve_primary_speaker(segment)
    try:
        primary_span = float(distribution.get(primary, 0.0))
    except (TypeError, ValueError):
        primary_span = 0.0
    if primary_span <= 0.0:
        return end, False
    trimmed_end = min(end, start + primary_span)
    trimmed = trimmed_end < (end - 0.05)
    return trimmed_end, trimmed


def _collect_segments(
    segments: Iterable[Mapping[str, object]],
    *,
    min_duration: float,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        trimmed_end, trimmed = _trim_segment(segment, start_f, end_f)
        duration = max(trimmed_end - start_f, 0.0)
        if duration <= 0.0:
            continue
        if duration < min_duration and not trimmed:
            continue
        grouped[_cluster_key(segment)].append(
            {
                "start": start_f,
                "end": trimmed_end,
                "original_end": end_f,
                "duration": duration,
                "original_duration": end_f - start_f,
                "trimmed": trimmed,
                "segment": segment,
            }
        )
    return grouped


def _write_snippet(
    clip: AudioClip,
    start: float,
    end: float,
    output_dir: Path,
    file_stem: str,
    index: int,
) -> Path:
    snippet = extract_segment(clip, start, end)
    path = output_dir / f"{file_stem}_{index:02d}.wav"
    save_audio(snippet, path, subtype="PCM_16", always_mono=True)
    return path


def main() -> int:
    args = _parse_args()
    audio_path, transcript_path, show_config = _resolve_paths(args)

    audio_path = audio_path.expanduser().resolve()
    transcript_path = transcript_path.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not transcript_path.exists():
        raise SystemExit(f"Transcript JSON not found: {transcript_path}")

    output_dir_root = args.output_dir.expanduser().resolve()
    episode_output_dir = output_dir_root / args.episode_id
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading audio from {audio_path}")
    clip = load_audio(audio_path, mono=True)
    print(f"[INFO] Audio duration: {clip.duration_seconds:.1f}s at {clip.sample_rate}Hz")

    segments = _load_transcript_segments(transcript_path)
    grouped = _collect_segments(segments, min_duration=args.min_duration)
    if not grouped:
        raise SystemExit("No segments found that meet the minimum duration criteria.")

    summary: dict[str, dict[str, object]] = {}

    for cluster_id, entries in sorted(grouped.items()):
        entries_sorted = sorted(entries, key=lambda item: item["duration"], reverse=True)
        limit = args.snippets_per_cluster if args.snippets_per_cluster > 0 else len(entries_sorted)
        selected = entries_sorted[:limit]

        cluster_dir = episode_output_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)

        snippet_entries: list[dict[str, object]] = []
        trimmed_count = 0
        for idx, entry in enumerate(selected, start=1):
            start = entry["start"]
            end = entry["end"]
            trimmed_flag = bool(entry.get("trimmed"))
            if trimmed_flag:
                trimmed_count += 1
            path = _write_snippet(
                clip,
                start,
                end,
                cluster_dir,
                f"{cluster_id.replace(' ', '_')}",
                idx,
            )
            snippet_entries.append(
                {
                    "id": idx,
                    "audio_path": str(path),
                    "start": start,
                    "end": end,
                    "duration": entry["duration"],
                    "original_end": entry["original_end"],
                    "original_duration": entry["original_duration"],
                    "trimmed": trimmed_flag,
                    "text": entry["segment"].get("text"),
                }
            )

        total_duration = sum(item["duration"] for item in entries)
        timestamp = datetime.now(ZoneInfo("UTC")).isoformat()

        summary[cluster_id] = {
            "snippets": snippet_entries,
            "total_segments": len(entries),
            "total_duration": float(total_duration),
            "cast_candidates": show_config.get("cast_members", []),
            "metadata": {
                "created": timestamp,
                "updated": timestamp,
                "trimmed_snippets": trimmed_count,
            },
        }

        print(
            f"[INFO] Exported {len(snippet_entries)} snippet(s) for {cluster_id} -> {cluster_dir}"
        )

    export_time = datetime.now(ZoneInfo("UTC")).isoformat()
    summary["_meta"] = {
        "created": export_time,
        "updated": export_time,
        "episode_id": args.episode_id,
        "snippets_per_cluster": args.snippets_per_cluster,
        "min_duration_seconds": args.min_duration,
    }

    summary_path = episode_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Summary written to {summary_path}")
    print(
        "[INFO] Review the snippets, decide on the speaker names, then run scripts/seed_voice_bank.py with the chosen mappings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
