#!/usr/bin/env python3
"""Benchmark ClearerVoice Studio against the default resemble-enhance stack."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import sys

# Ensure project modules are importable when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from show_scribe.pipelines.audio_preprocessing import AudioPreprocessor  # noqa: E402
from show_scribe.pipelines.orchestrator import build_default_context  # noqa: E402
from show_scribe.pipelines.asr import (  # noqa: E402
    TranscriptionResult,
    build_hybrid_transcriber,
)


@dataclass(slots=True)
class BenchmarkResult:
    provider: str
    runtime_seconds: float
    artifacts: Mapping[str, str | None]
    bytes_written: Mapping[str, int]
    wer: float | None
    word_count: int | None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "runtime_seconds": self.runtime_seconds,
            "artifacts": dict(self.artifacts),
            "bytes_written": dict(self.bytes_written),
            "wer": self.wer,
            "word_count": self.word_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_clip", type=Path, help="Path to a WAV/MP3/MP4 clip (5–10 minutes ideal)"
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Config environment to load (default: dev)",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        default=None,
        help="Optional plain-text reference transcript for WER scoring.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=PROJECT_ROOT / "results" / "bench_clearervoice.json",
        help="Where to store the JSON summary (default: results/bench_clearervoice.json).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Preserve provider-specific temp directories for inspection.",
    )
    return parser.parse_args()


def calculate_wer(reference: str | None, hypothesis: str | None) -> float | None:
    if not reference or not hypothesis:
        return None

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return None

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    d = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        d[i, 0] = i
    for j in range(cols):
        d[0, j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[-1, -1] / len(ref_words))


def flatten_transcript(result: TranscriptionResult | None) -> str:
    if result is None:
        return ""
    return " ".join(segment.text.strip() for segment in result.segments).strip()


def _stats_for_path(path: Path | None) -> int:
    if not path:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _resolve_reference(ref_path: Path | None) -> str | None:
    if not ref_path:
        return None
    try:
        return ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def run_provider(
    provider: str,
    input_clip: Path,
    env: str,
    *,
    temp_root: Path,
    reference_text: str | None,
) -> BenchmarkResult:
    overrides = {
        "audio_preprocessing": {
            "enable": True,
            "retain_intermediates": True,
            "vocal_separation": {"enable": True},
            "enhancement": {"enable": True, "provider": provider},
        }
    }

    context = build_default_context(env, overrides=overrides)
    preprocessor = AudioPreprocessor(context.config)

    provider_dir = temp_root / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    artifacts = preprocessor.preprocess(str(input_clip), output_dir=str(provider_dir))
    runtime = time.perf_counter() - start

    artifacts_map: Dict[str, str | None] = {
        "final_audio": str(artifacts.final_audio) if artifacts.final_audio else None,
        "vocals_audio": str(artifacts.vocals_audio) if artifacts.vocals_audio else None,
        "enhanced_audio": str(artifacts.enhanced_audio) if artifacts.enhanced_audio else None,
        "enhanced_mix_audio": (
            str(artifacts.enhanced_mix_audio) if artifacts.enhanced_mix_audio else None
        ),
        "report": str(artifacts.report_path),
    }

    bytes_written = {
        "final_audio": _stats_for_path(artifacts.final_audio),
        "vocals_audio": _stats_for_path(artifacts.vocals_audio),
        "enhanced_audio": _stats_for_path(artifacts.enhanced_audio),
        "enhanced_mix_audio": _stats_for_path(artifacts.enhanced_mix_audio),
    }

    wer_value: float | None = None
    word_count: int | None = None
    if reference_text:
        transcriber = build_hybrid_transcriber(context.config.get("whisper", {}), context.paths)
        transcript = transcriber.transcribe(str(artifacts.final_audio))
        hypothesis = flatten_transcript(transcript)
        wer_value = calculate_wer(reference_text, hypothesis)
        word_count = len(reference_text.split()) if reference_text else None

    return BenchmarkResult(
        provider=provider,
        runtime_seconds=runtime,
        artifacts=artifacts_map,
        bytes_written=bytes_written,
        wer=wer_value,
        word_count=word_count,
    )


def main() -> int:
    args = parse_args()
    input_clip = args.input_clip.expanduser().resolve()
    if not input_clip.exists():
        print(f"Input clip not found: {input_clip}")
        return 1

    reference_text = _resolve_reference(args.reference_transcript)
    results_dir = args.results_file.expanduser().resolve().parent
    results_dir.mkdir(parents=True, exist_ok=True)

    temp_root = PROJECT_ROOT / "outputs" / "benchmarks" / "clearervoice"
    if temp_root.exists() and not args.keep_intermediates:
        shutil.rmtree(temp_root, ignore_errors=True)

    temp_root.mkdir(parents=True, exist_ok=True)

    providers = ["resemble", "clearervoice"]
    benchmark_results: dict[str, BenchmarkResult] = {}
    for provider in providers:
        try:
            result = run_provider(
                provider,
                input_clip,
                args.env,
                temp_root=temp_root,
                reference_text=reference_text,
            )
        except Exception as exc:
            print(f"Provider '{provider}' failed: {exc}")
            if provider == "clearervoice":
                print(
                    "Install ClearerVoice with `pip install clearvoice` and re-run the benchmark."
                )
            return 2
        else:
            benchmark_results[provider] = result
            print(
                f"{provider:>11}: {result.runtime_seconds:6.2f}s | final={result.bytes_written['final_audio'] / 1_048_576:5.1f} MB"
                + (f" | WER={result.wer:.3f}" if result.wer is not None else "")
            )

    resemble_result = benchmark_results.get("resemble")
    clearervoice_result = benchmark_results.get("clearervoice")

    comparison: Dict[str, Any] = {}
    if resemble_result and clearervoice_result:
        comparison["runtime_delta_seconds"] = (
            clearervoice_result.runtime_seconds - resemble_result.runtime_seconds
        )
        if resemble_result.wer is not None and clearervoice_result.wer is not None:
            comparison["wer_delta"] = clearervoice_result.wer - resemble_result.wer

    payload = {
        "input_clip": str(input_clip),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": args.env,
        "reference_transcript": (
            str(args.reference_transcript) if args.reference_transcript else None
        ),
        "providers": {key: result.as_dict() for key, result in benchmark_results.items()},
        "comparison": comparison,
    }

    args.results_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved results to {args.results_file}")

    if not args.keep_intermediates:
        shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
