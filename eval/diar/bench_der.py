#!/usr/bin/env python3
"""Benchmark harness for diarization error metrics across presets."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from show_scribe.config.load import ConfigError
from show_scribe.config.load import load_config as load_env_config
from show_scribe.pipelines.diarization.pyannote_pipeline import (
    PyannoteDiarizer,
    PyannoteNotAvailableError,
    build_pyannote_diarizer,
)

LOGGER = logging.getLogger("bench_der")

DEFAULT_CONFIGS = [
    "configs/dev.yaml",
    "configs/prod.yaml",
    "configs/reality_tv.yaml",
]
RESULTS_FILENAME = "der_results.csv"
SUMMARY_FILENAME = "README.md"
OUTLIER_DELTA = 0.05

try:  # Optional metric
    from pyannote.metrics.diarization import JaccardErrorRate
except ImportError:  # pragma: no cover - dependency optional
    JaccardErrorRate = None


@dataclass(frozen=True)
class BenchmarkItem:
    """Single diarization test case pulled from the manifest."""

    clip_id: str
    audio_path: Path
    reference: Annotation
    overlap_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark diarization presets with DER/JER metrics."
    )
    parser.add_argument(
        "--manifest",
        default="eval/diar/manifest.json",
        help="Path to manifest JSON describing reference annotations.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config environments or YAML files to benchmark.",
    )
    parser.add_argument(
        "--results-dir",
        default="eval/diar/results",
        help="Output directory for CSV/README summaries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    manifest_path = Path(args.manifest)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_items = load_manifest(manifest_path)
    if not benchmark_items:
        raise SystemExit(f"No entries found in manifest {manifest_path}.")

    configs: list[tuple[str, dict[str, Any]]] = []
    for identifier in args.configs:
        config_label, payload = resolve_config(identifier)
        configs.append((config_label, payload))

    diarizers: dict[str, PyannoteDiarizer] = {}
    for config_label, payload in configs:
        LOGGER.info("Preparing diarizer for %s", config_label)
        diarizers[config_label] = build_pyannote_diarizer(payload)

    records: list[dict[str, Any]] = []
    for config_label, _ in configs:
        diarizer = diarizers[config_label]
        for item in benchmark_items:
            LOGGER.info("Running %s on %s", config_label, item.clip_id)
            record = evaluate_clip(diarizer, config_label, item)
            records.append(record)

    write_csv(records, results_dir / RESULTS_FILENAME)
    write_summary(records, results_dir / SUMMARY_FILENAME)
    LOGGER.info("Wrote %s entries to %s", len(records), results_dir)


def resolve_config(identifier: str) -> tuple[str, dict[str, Any]]:
    """Load a Show-Scribe configuration from env name or YAML path."""
    path = Path(identifier)
    if path.suffix in {".yaml", ".yml"}:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        env_name = path.stem
        payload = load_env_config(env_name, config_dir=path.parent)
        return identifier, payload
    payload = load_env_config(identifier)
    return identifier, payload


def load_manifest(path: Path) -> list[BenchmarkItem]:
    """Return benchmark items described in the manifest JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a list of objects.")

    items: list[BenchmarkItem] = []
    for entry in entries:
        clip_id = str(entry["clip_id"])
        audio_path = Path(entry["audio"])
        reference = load_reference_annotation(entry)
        overlap_ratio = compute_overlap_ratio(reference)
        items.append(BenchmarkItem(clip_id, audio_path, reference, overlap_ratio))
    return items


def load_reference_annotation(entry: Mapping[str, Any]) -> Annotation:
    """Return a pyannote Annotation built from manifest reference data."""
    reference_block = dict(entry.get("reference") or {})
    # Support legacy top-level keys for convenience
    if "ref_rttm" in entry and "rttm" not in reference_block:
        reference_block["rttm"] = entry["ref_rttm"]
    if "ref_segments" in entry and "segments" not in reference_block:
        reference_block["segments"] = entry["ref_segments"]

    if "rttm" in reference_block:
        path = Path(reference_block["rttm"])
        return annotation_from_rttm(path)
    if "segments" in reference_block:
        return annotation_from_segments(reference_block["segments"])
    raise ValueError(f"Manifest entry missing reference data: {entry}")


def annotation_from_segments(segments: Sequence[Mapping[str, Any]]) -> Annotation:
    """Convert inline segments to a pyannote Annotation."""
    annotation = Annotation()
    for idx, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        speaker = str(segment.get("speaker") or f"speaker_{idx}")
        annotation[Segment(start, end)] = speaker
    return annotation


def annotation_from_rttm(path: Path) -> Annotation:
    """Load RTTM file and convert to Annotation."""
    if not path.exists():
        raise FileNotFoundError(f"Reference RTTM not found: {path}")
    annotation = Annotation(uri=path.stem)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            if end <= start:
                continue
            annotation[Segment(start, end)] = speaker
    return annotation


def evaluate_clip(
    diarizer: PyannoteDiarizer,
    config_label: str,
    item: BenchmarkItem,
) -> dict[str, Any]:
    """Run diarization on a single clip and compute DER/JER."""
    der_value = math.nan
    jer_value = math.nan
    speakers_detected = 0

    if not item.audio_path.exists():
        LOGGER.warning("Audio clip missing for %s (%s)", item.clip_id, item.audio_path)
        return {
            "config": config_label,
            "clip_id": item.clip_id,
            "DER": der_value,
            "JER": jer_value,
            "speakers_detected": speakers_detected,
            "overlap_ratio": item.overlap_ratio,
        }

    try:
        result = diarizer.diarize(item.audio_path)
        hypothesis = segments_to_annotation(result.segments, uri=item.clip_id)
        speakers_detected = len(hypothesis.labels())
        der_value = compute_der(item.reference, hypothesis)
        jer_value = compute_jer(item.reference, hypothesis)
    except (PyannoteNotAvailableError, ConfigError) as exc:
        LOGGER.error("Failed to run diarization for %s: %s", item.clip_id, exc)
    except Exception:  # pragma: no cover - runtime safety
        LOGGER.exception("Unexpected diarization failure for %s", item.clip_id)
    return {
        "config": config_label,
        "clip_id": item.clip_id,
        "DER": der_value,
        "JER": jer_value,
        "speakers_detected": speakers_detected,
        "overlap_ratio": item.overlap_ratio,
    }


def segments_to_annotation(segments: Iterable[Any], *, uri: str | None = None) -> Annotation:
    """Build an Annotation from PyannoteDiarizer segments."""
    annotation = Annotation(uri=uri)
    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        speaker = str(getattr(segment, "speaker", "speaker"))
        annotation[Segment(start, end)] = speaker
    return annotation


def compute_der(reference: Annotation, hypothesis: Annotation) -> float:
    metric = DiarizationErrorRate()
    return float(metric(reference, hypothesis))


def compute_jer(reference: Annotation, hypothesis: Annotation) -> float:
    if JaccardErrorRate is None:
        return math.nan
    metric = JaccardErrorRate()
    return float(metric(reference, hypothesis))


def compute_overlap_ratio(annotation: Annotation) -> float:
    try:
        total = annotation.get_timeline().duration()
    except Exception:  # pragma: no cover - defensive
        return 0.0
    if not total:
        return 0.0
    try:
        overlap_timeline = annotation.get_overlap()
        overlap = overlap_timeline.duration()
    except Exception:
        overlap = 0.0
    return float(overlap / total) if total else 0.0


def write_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Persist per-clip results to CSV."""
    columns = ["config", "clip_id", "DER", "JER", "speakers_detected", "overlap_ratio"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for record in records:
            row = [
                str(record["config"]),
                str(record["clip_id"]),
                format_float(record["DER"]),
                format_float(record["JER"]),
                str(record["speakers_detected"]),
                format_float(record["overlap_ratio"]),
            ]
            handle.write(",".join(row) + "\n")


def write_summary(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write README summary with per-config averages and outliers."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    summary_stats = aggregate_by_config(records)
    outliers = detect_outliers(records, summary_stats)

    lines: list[str] = ["# DER Benchmark Results", "", f"Generated: {timestamp}", ""]
    if not summary_stats:
        lines.append("No successful diarization runs recorded yet.")
    else:
        lines.append("| Config | Clips | Mean DER | Mean JER | Mean Speakers |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for config, stats in summary_stats.items():
            lines.append(
                f"| {config} | {stats['clips']} | "
                f"{format_float(stats['mean_der'])} | "
                f"{format_float(stats['mean_jer'])} | "
                f"{format_float(stats['mean_speakers'])} |"
            )
    lines.append("")
    lines.append(
        f"Outliers are clips where DER exceeded (mean + {OUTLIER_DELTA:.2f}) for the same config."
    )
    lines.append("")
    if outliers:
        lines.append("| Config | Clip | DER | Δ vs Mean |")
        lines.append("| --- | --- | ---: | ---: |")
        for entry in outliers:
            lines.append(
                f"| {entry['config']} | {entry['clip_id']} | "
                f"{format_float(entry['DER'])} | {format_float(entry['delta'])} |"
            )
    else:
        lines.append("No clip-level outliers detected.")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def aggregate_by_config(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Compute mean DER/JER per config."""
    aggregated: dict[str, dict[str, Any]] = {}
    for record in records:
        config = str(record["config"])
        der = record["DER"]
        jer = record["JER"]
        speakers = record["speakers_detected"]
        if math.isnan(der):
            continue
        slot = aggregated.setdefault(
            config,
            {
                "clips": 0,
                "sum_der": 0.0,
                "count_der": 0,
                "sum_jer": 0.0,
                "count_jer": 0,
                "sum_speakers": 0.0,
            },
        )
        slot["clips"] += 1
        slot["sum_der"] += der
        slot["count_der"] += 1
        slot["sum_speakers"] += speakers
        if not math.isnan(jer):
            slot["sum_jer"] += jer
            slot["count_jer"] += 1

    summary: dict[str, dict[str, float | int]] = {}
    for config, stats in aggregated.items():
        summary[config] = {
            "clips": int(stats["clips"]),
            "mean_der": stats["sum_der"] / max(stats["count_der"], 1),
            "mean_jer": (stats["sum_jer"] / stats["count_jer"] if stats["count_jer"] else math.nan),
            "mean_speakers": stats["sum_speakers"] / max(stats["count_der"], 1),
        }
    return summary


def detect_outliers(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, float | int]],
) -> list[dict[str, Any]]:
    """Return clip rows where DER is meaningfully above the config mean."""
    outliers: list[dict[str, Any]] = []
    for record in records:
        config = str(record["config"])
        der = record["DER"]
        if math.isnan(der):
            continue
        mean = summary.get(config, {}).get("mean_der")
        if mean is None or math.isnan(mean):
            continue
        threshold = mean + OUTLIER_DELTA
        if der > threshold:
            outliers.append(
                {
                    "config": config,
                    "clip_id": record["clip_id"],
                    "DER": der,
                    "delta": der - mean,
                }
            )
    return outliers


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.4f}"


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
