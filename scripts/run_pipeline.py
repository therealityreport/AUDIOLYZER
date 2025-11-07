"""CLI driver to process a single episode end-to-end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from pipeline.audio_ops import AudioPreprocessingError, ensure_ffmpeg_available, extract_and_enhance
from show_scribe.pipelines.orchestrator import (
    JobStatus,
    PipelineContext,
    PipelineJob,
    PipelineOrchestrator,
    build_default_context,
)
from show_scribe.storage.paths import PathsConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source media file.")
    parser.add_argument(
        "--episode-id",
        required=True,
        help="Identifier used for output directories (e.g., RHOBH_S05E01_fullscene).",
    )
    parser.add_argument("--show-config", help="Optional explicit path to a show_config.json file.")
    parser.add_argument(
        "--preset",
        default="configs/reality_tv.yaml",
        help="Name or path of the audio preprocessing preset (default: configs/reality_tv.yaml).",
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        help="Episode directory for preprocessing artifacts (defaults to show configuration).",
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Configuration environment to load (default: dev).",
    )
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Force-enable audio preprocessing regardless of configuration.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Force-disable audio preprocessing regardless of configuration.",
    )
    parser.add_argument(
        "--allow-fallback-audio",
        action="store_true",
        help="Continue with extracted audio when preprocessing fails.",
    )
    parser.set_defaults(preprocess=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent pipeline workers (default: 1).",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=4,
        help="Maximum number of queued jobs before rejecting new submissions (default: 4).",
    )
    return parser.parse_args(argv)


def _progress_printer(job: PipelineJob) -> None:
    percent = f"{job.progress * 100:.0f}%"
    stage = job.current_stage or "<initialising>"
    message = job.message or ""
    print(f"[{job.job_id}] {stage:<24} {percent:>4} {message}")


def _load_context(env: str, *, overrides: dict[str, object] | None = None) -> PipelineContext:
    return build_default_context(env, overrides=overrides)


def _resolve_show_config(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"show_config file not found: {candidate}")
    return candidate


def _infer_show_config_path(episode_id: str, paths: PathsConfig) -> Path | None:
    shows_root = paths.data_root / "shows"
    if not shows_root.exists():
        return None
    slug_candidate = episode_id.split("_", 1)[0].lower()
    for child in shows_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() == slug_candidate:
            candidate = child / "show_config.json"
            if candidate.exists():
                return candidate
    return None


def _load_show_config_data(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Show config at {path} must be a JSON object.")
    return data


def _select_episode_root(
    paths: PathsConfig,
    show_config_path: Path,
    show_config: Mapping[str, Any] | None,
) -> Path:
    custom_paths = show_config.get("paths") if show_config else None
    if isinstance(custom_paths, Mapping):
        raw_root = custom_paths.get("show_root")
        if isinstance(raw_root, str) and raw_root.strip():
            return Path(raw_root).expanduser().resolve()

    slug = str(show_config.get("show_slug") if show_config else "")
    if not slug:
        slug = show_config_path.parent.name
    return paths.show_root(slug)


def _derive_episode_dir(
    paths: PathsConfig,
    episode_id: str,
    *,
    explicit_dir: Path | None,
    show_config_path: Path | None,
    show_config: Mapping[str, Any] | None,
) -> Path:
    if explicit_dir is not None:
        return explicit_dir.expanduser().resolve()
    if show_config_path is None:
        raise ValueError(
            "Episode directory cannot be derived without --episode-dir or --show-config."
        )

    root = _select_episode_root(paths, show_config_path, show_config)
    return (root / "episodes" / episode_id).resolve()


def _resolve_preset(value: str) -> Path:
    if not value or value == "reality_tv":
        candidate = Path("configs/reality_tv.yaml")
    else:
        candidate = Path(value)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Preset file not found: {candidate}")
    return candidate


def _resolve_preprocessing_setting(
    cli_override: bool | None,
    config: Mapping[str, Any],
) -> tuple[bool, str]:
    config_enable = False
    audio_section = config.get("audio_preprocessing")
    if isinstance(audio_section, Mapping):
        config_enable = bool(audio_section.get("enable", False))

    if cli_override is False:
        return False, "--no-preprocess"
    if cli_override is True:
        return True, "--preprocess"
    return config_enable, "config"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        ensure_ffmpeg_available()
    except RuntimeError as exc:  # pragma: no cover - environment guard
        print(str(exc), file=sys.stderr)
        return 2

    preset_path: Path
    try:
        preset_path = _resolve_preset(args.preset)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cli_override = None
    if args.preprocess is not None:
        cli_override = bool(args.preprocess)

    overrides: dict[str, object] | None = None
    if cli_override is not None:
        overrides = {"audio_preprocessing": {"enable": cli_override}}

    context = _load_context(args.env, overrides=overrides)

    resolved_setting, source = _resolve_preprocessing_setting(cli_override, context.config)
    state_text = "enabled" if resolved_setting else "disabled"
    print(f"Audio preprocessing: {state_text} (source={source}).")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    show_config_path = _resolve_show_config(args.show_config)
    show_config_data: Mapping[str, Any] | None = None
    if show_config_path is None:
        show_config_path = _infer_show_config_path(args.episode_id, context.paths)
    if show_config_path is not None:
        try:
            show_config_data = _load_show_config_data(show_config_path)
        except (OSError, ValueError) as exc:
            print(f"Failed to load show config: {exc}", file=sys.stderr)
            return 2

    explicit_episode_dir = args.episode_dir
    try:
        episode_dir = _derive_episode_dir(
            context.paths,
            args.episode_id,
            explicit_dir=explicit_episode_dir,
            show_config_path=show_config_path,
            show_config=show_config_data,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    episode_dir.mkdir(parents=True, exist_ok=True)

    pipeline_input = input_path
    job_metadata: dict[str, Any] = {"episode_dir_override": str(episode_dir)}

    if resolved_setting:
        try:
            outputs = extract_and_enhance(input_path, episode_dir, preset_path)
        except AudioPreprocessingError as exc:
            print(f"Audio preprocessing failed: {exc}", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr.strip(), file=sys.stderr)
            fallback_candidate = (
                exc.artifacts.get("extracted") if hasattr(exc, "artifacts") else None
            )
            if isinstance(fallback_candidate, Path):
                fallback_path = fallback_candidate.resolve()
            else:
                fallback_path = episode_dir / "audio_extracted.wav"
            if args.allow_fallback_audio and fallback_path.exists():
                pipeline_input = fallback_path
                print(f"Continuing with fallback audio: {fallback_path}")
            else:
                if args.allow_fallback_audio:
                    print("Fallback audio unavailable; aborting.", file=sys.stderr)
                return 4
        else:
            enhanced = (
                outputs.get("enhanced_vocals")
                or outputs.get("enhanced")
                or outputs.get("enhanced_mix")
            )
            if not enhanced or not Path(enhanced).exists():
                print("Enhanced audio not produced; aborting.", file=sys.stderr)
                return 4
            pipeline_input = Path(enhanced).resolve()
            report_path = outputs.get("report")
            if report_path:
                print(f"Preprocessing report: {report_path}")
            print(f"Using enhanced audio: {pipeline_input}")

    orchestrator = PipelineOrchestrator(
        context=context,
        max_workers=max(args.max_workers, 1),
        max_queue=max(args.max_queue, 1),
    )

    try:
        job = orchestrator.submit(
            pipeline_input,
            episode_id=args.episode_id,
            show_config_path=show_config_path,
            metadata=job_metadata,
            progress_handler=_progress_printer,
        )
    except Exception as exc:  # pragma: no cover - CLI bridge
        orchestrator.shutdown()
        print(f"Failed to enqueue job: {exc}", file=sys.stderr)
        return 2

    print(f"Submitted job {job.job_id} for episode '{job.episode_id}'.")

    orchestrator.wait(job.job_id)
    orchestrator.shutdown()

    final_job = orchestrator.get_job(job.job_id)
    if final_job is None:
        print("Unexpected error: job disappeared from orchestrator.", file=sys.stderr)
        return 3

    if final_job.status is JobStatus.COMPLETED:
        print("Pipeline completed successfully.")
        return 0

    print(f"Pipeline failed: {final_job.message}", file=sys.stderr)
    if final_job.error:
        print(final_job.error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
