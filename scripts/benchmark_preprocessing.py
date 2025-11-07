#!/usr/bin/env python3
"""
Benchmark audio preprocessing impact on transcription accuracy.

This script extracts, preprocesses, and transcribes an episode with multiple
configurations so operators can quantify Word Error Rate (WER) improvements
and latency trade-offs before enabling preprocessing in production.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Ensure the project sources are importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from show_scribe.pipelines.asr import TranscriptionResult, build_hybrid_transcriber
from show_scribe.pipelines.extract_audio import AudioExtractionConfig, AudioExtractor
from show_scribe.pipelines.orchestrator import build_default_context
from show_scribe.storage.naming import EpisodeDescriptor


@dataclass(slots=True)
class Scenario:
    """Benchmark scenario definition."""

    name: str
    description: str
    overrides: dict[str, Any]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="baseline",
        description="No preprocessing (raw audio).",
        overrides={"audio_preprocessing": {"enable": False}},
    ),
    Scenario(
        name="separation",
        description="Vocal separation only.",
        overrides={
            "audio_preprocessing": {
                "enable": True,
                "vocal_separation": {"enable": True},
                "enhancement": {"enable": False},
            }
        },
    ),
    Scenario(
        name="enhancement",
        description="Enhancement only (denoise + dereverb).",
        overrides={
            "audio_preprocessing": {
                "enable": True,
                "vocal_separation": {"enable": False},
                "enhancement": {"enable": True},
            }
        },
    ),
    Scenario(
        name="both",
        description="Reality TV preset (separation + enhancement).",
        overrides={
            "audio_preprocessing": {
                "enable": True,
                "vocal_separation": {"enable": True, "model": "htdemucs"},
                "enhancement": {
                    "enable": True,
                    "denoise": True,
                    "dereverb": True,
                    "lambd": 0.7,
                    "tau": 0.6,
                },
            }
        },
    ),
    Scenario(
        name="aggressive",
        description="Aggressive enhancement (for extremely noisy content).",
        overrides={
            "audio_preprocessing": {
                "enable": True,
                "vocal_separation": {"enable": True, "model": "htdemucs"},
                "enhancement": {
                    "enable": True,
                    "denoise": True,
                    "dereverb": True,
                    "lambd": 0.9,
                    "tau": 0.8,
                },
            }
        },
    ),
]

_EPISODE_CODE_RE = re.compile(r"S(?P<season>\d{1,2})E(?P<episode>\d{1,2})", re.IGNORECASE)


def calculate_wer(reference: str | None, hypothesis: str | None) -> float | None:
    """
    Calculate Word Error Rate between reference and hypothesis.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong).
        Returns None when either text is missing.
    """
    if not reference or not hypothesis:
        return None

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return None

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    d = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[-1][-1] / len(ref_words)


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _split_episode_id(raw: str) -> tuple[str, int, int, str | None]:
    """Best-effort parsing of a pipeline episode identifier."""
    match = _EPISODE_CODE_RE.search(raw)
    if not match:
        return raw or "benchmark", 0, 0, None

    prefix = raw[: match.start()].rstrip("_- ")
    remainder = raw[match.end() :].lstrip("_- ")
    season = int(match.group("season"))
    episode = int(match.group("episode"))
    return prefix or "benchmark", season, episode, remainder or None


def build_descriptor(
    episode_path: Path, base_episode_id: str | None, scenario_slug: str
) -> EpisodeDescriptor:
    """Construct a unique descriptor per scenario to isolate artifacts."""
    if base_episode_id:
        show_name_raw, season, episode, remainder = _split_episode_id(base_episode_id)
    else:
        show_name_raw = episode_path.stem or "benchmark"
        season = 0
        episode = 0
        remainder = None

    show_name = f"benchmark_{slugify(show_name_raw) or 'episode'}"
    variant_tokens = [slugify(token) for token in (remainder, scenario_slug) if token]
    variant = "_".join(token for token in variant_tokens if token) or scenario_slug
    return EpisodeDescriptor(
        show_name=show_name, season=season, episode=episode, variant=variant or None
    )


def flatten_transcript(result: TranscriptionResult) -> str:
    """Return a plain-text transcript from Faster-Whisper segments."""
    return " ".join(segment.text.strip() for segment in result.segments).strip()


def run_scenario(
    episode_path: Path,
    base_episode_id: str | None,
    env: str,
    scenario: Scenario,
    reference_transcript: str | None,
) -> dict[str, Any]:
    """Execute extraction + transcription for a single scenario."""
    scenario_slug = slugify(scenario.name)
    descriptor = build_descriptor(episode_path, base_episode_id, scenario_slug)

    context = build_default_context(env, overrides=scenario.overrides)
    paths = context.paths
    episode_dir = paths.episode_directory(descriptor.show_name, descriptor.episode_id)
    if episode_dir.exists():
        shutil.rmtree(episode_dir)

    extraction_config = AudioExtractionConfig.from_config(context.config)
    extractor = AudioExtractor(paths, extraction_config, full_config=dict(context.config))

    extraction_start = time.perf_counter()
    extraction_result = extractor.extract(episode_path, descriptor)
    extraction_seconds = time.perf_counter() - extraction_start

    preprocessing_report = extraction_result.preprocessing_report or {}
    preprocessing_seconds = (
        float(sum(preprocessing_report.get("timings_seconds", {}).values()))
        if preprocessing_report
        else 0.0
    )
    preprocessing_report_path = (
        str(extraction_result.preprocessing_report_path)
        if extraction_result.preprocessing_report_path
        else None
    )

    transcriber = build_hybrid_transcriber(context.config, paths)
    transcription_start = time.perf_counter()
    transcription = transcriber.transcribe(extraction_result.audio_path)
    transcription_seconds = time.perf_counter() - transcription_start

    transcript_text = flatten_transcript(transcription)
    wer = calculate_wer(reference_transcript, transcript_text)

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "preprocessing_enabled": preprocessing_report.get("preprocessing_enabled", False),
        "steps": preprocessing_report.get("steps_applied", []),
        "audio_path": str(extraction_result.audio_path),
        "preprocessing_report_path": preprocessing_report_path,
        "extraction_seconds": extraction_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "transcription_seconds": transcription_seconds,
        "total_seconds": extraction_seconds + transcription_seconds,
        "wer": wer,
        "transcript_text": transcript_text,
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def print_summary(results: Iterable[dict[str, Any]]) -> None:
    """Print a human-friendly summary table."""
    results = list(results)
    if not results:
        print("No benchmark results to display.")
        return

    headers = [
        "Scenario",
        "Steps",
        "Extraction (s)",
        "Preproc (s)",
        "Transcription (s)",
        "Total (s)",
        "WER",
        "Δ WER vs Baseline",
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))

    for result in results:
        wer = result["wer"]
        delta = result.get("wer_delta_vs_baseline_pct")
        row = [
            result["scenario"],
            ", ".join(result["steps"]) if result["steps"] else "—",
            _format_seconds(result["extraction_seconds"]),
            _format_seconds(result["preprocessing_seconds"]),
            _format_seconds(result["transcription_seconds"]),
            _format_seconds(result["total_seconds"]),
            f"{wer:.3f}" if wer is not None else "N/A",
            _format_percent(delta),
        ]
        print("| " + " | ".join(row) + " |")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", help="Path to the source media file to benchmark.")
    parser.add_argument(
        "--reference-transcript",
        help="Optional path to a ground-truth transcript (plain text) for WER calculation.",
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Configuration environment to load (default: dev).",
    )
    parser.add_argument(
        "--episode-id",
        help="Optional pipeline episode identifier (used for naming outputs).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Subset of scenarios to run. Defaults to all.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write benchmark results as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episode_path = Path(args.episode_path).expanduser().resolve()
    if not episode_path.exists():
        print(f"Episode not found: {episode_path}", file=sys.stderr)
        return 2

    reference_text = None
    if args.reference_transcript:
        ref_path = Path(args.reference_transcript).expanduser().resolve()
        if not ref_path.exists():
            print(f"Reference transcript not found: {ref_path}", file=sys.stderr)
            return 3
        reference_text = ref_path.read_text(encoding="utf-8")

    selected_names = (
        set(args.scenarios) if args.scenarios else {scenario.name for scenario in SCENARIOS}
    )
    scenarios = [scenario for scenario in SCENARIOS if scenario.name in selected_names]

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario.name} ===")
        print(f"{scenario.description}")
        result = run_scenario(
            episode_path=episode_path,
            base_episode_id=args.episode_id,
            env=args.env,
            scenario=scenario,
            reference_transcript=reference_text,
        )
        results.append(result)
        print(
            f"Completed {scenario.name}: total={result['total_seconds']:.2f}s, "
            f"WER={'N/A' if result['wer'] is None else f'{result['wer']:.3f}'}"
        )

    baseline_result = next((item for item in results if item["scenario"] == "baseline"), None)
    baseline_wer = baseline_result["wer"] if baseline_result else None
    for result in results:
        if baseline_wer is not None and result["wer"] is not None:
            result["wer_delta_vs_baseline_pct"] = (
                (baseline_wer - result["wer"]) / baseline_wer * 100
            )
        else:
            result["wer_delta_vs_baseline_pct"] = None

    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved benchmark results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
