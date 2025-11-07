"""Lightweight job orchestrator for coordinating pipeline runs."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from enum import Enum, auto
from pathlib import Path
from queue import Full, Queue
from typing import Any

from show_scribe.config.load import load_config
from show_scribe.pipelines.alignment.align_asr_diar import AlignmentResult
from show_scribe.pipelines.asr import (
    TranscriptionOptions,
    TranscriptionResult,
    build_hybrid_transcriber,
)
from show_scribe.pipelines.diarization.pyannote_pipeline import (
    DiarizationResult,
    build_pyannote_diarizer,
)
from show_scribe.pipelines.extract_audio import (
    AudioExtractionConfig,
    AudioExtractor,
    AudioExtractionError,
)
from show_scribe.pipelines.transcript.builder import TranscriptBuilder, TranscriptDocument
from show_scribe.pipelines.transcript.export_json import build_transcript_payload
from show_scribe.pipelines.transcript.export_srt import render_srt
from show_scribe.pipelines.transcript.export_text import render_plain_text
from show_scribe.pipelines.speaker_id.voice_bank import (
    SpeakerIdentificationError,
    build_voice_bank_pipeline,
)
from show_scribe.pipelines.transcript.pipeline import TranscriptPipeline
from show_scribe.storage.naming import EpisodeDescriptor
from show_scribe.storage.paths import PathsConfig, build_paths
from show_scribe.utils.logging import get_logger
from show_scribe.utils.name_correction import NameCorrector
from show_scribe.utils import audio_io

LOGGER = get_logger(__name__)

StageRunner = Callable[["PipelineJob", "PipelineContext", "StageProgress"], str | None]


@dataclass(slots=True)
class StageProgress:
    """Helper for reporting fine-grained stage progress back to the orchestrator."""

    job: "PipelineJob"
    start_weight: float
    stage_weight: float
    total_weight: float
    _updated: bool = False

    def update(self, fraction: float, *, message: str | None = None) -> None:
        fraction = max(0.0, min(1.0, fraction))
        overall = (self.start_weight + self.stage_weight * fraction) / self.total_weight
        self.job.set_progress(overall, message=message)
        self._updated = True

    @property
    def updated(self) -> bool:
        return self._updated


_EPISODE_CODE_RE = re.compile(r"S(?P<season>\d{1,2})E(?P<episode>\d{1,2})", re.IGNORECASE)


def _parse_episode_descriptor(episode_id: str) -> tuple[int, int, str | None]:
    match = _EPISODE_CODE_RE.search(episode_id)
    if not match:
        raise ValueError(f"Unable to parse season/episode from episode_id '{episode_id}'.")
    season = int(match.group("season"))
    episode = int(match.group("episode"))
    remainder = episode_id[match.end() :].lstrip("_- ")
    return season, episode, remainder or None


def _load_show_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Show config at {path} must be a JSON object.")
    return data


def _infer_show_config_path(episode_id: str, paths: PathsConfig) -> Path | None:
    shows_root = paths.data_root / "shows"
    if not shows_root.exists():
        return None
    slug_candidate = episode_id.split("_", 1)[0]
    for child in shows_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() == slug_candidate.lower():
            candidate = child / "show_config.json"
            if candidate.exists():
                return candidate
    return None


class JobStatus(Enum):
    """Lifecycle states for a pipeline job."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass(slots=True)
class PipelineContext:
    """Runtime context shared across stages."""

    config: Mapping[str, Any]
    paths: PathsConfig
    environment: str


@dataclass(slots=True)
class StageDefinition:
    """A single pipeline stage definition."""

    name: str
    runner: StageRunner
    weight: float = 1.0


@dataclass(slots=True)
class PipelineJob:
    """Represents an episode processing job."""

    job_id: str
    input_path: Path
    episode_id: str
    show_config_path: Path | None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_stage: str | None = None
    message: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _completion_event: threading.Event = field(default_factory=threading.Event, init=False)
    _progress_handler: Callable[[PipelineJob], None] | None = None

    def mark(self, status: JobStatus, *, message: str | None = None) -> None:
        self.status = status
        if message is not None:
            self.message = message

    def set_progress(self, value: float, *, message: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, value))
        if message is not None:
            self.message = message
        if self._progress_handler is not None:
            self._progress_handler(self)

    def wait(self, timeout: float | None = None) -> bool:
        return self._completion_event.wait(timeout)


class PipelineOrchestrator:
    """Coordinates asynchronous execution of pipeline jobs."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        stages: Iterable[StageDefinition] | None = None,
        max_workers: int = 1,
        max_queue: int = 4,
    ) -> None:
        self._context = context
        self._stages = list(stages or DEFAULT_STAGES)
        self._total_weight = sum(stage.weight for stage in self._stages) or 1.0
        self._jobs: dict[str, PipelineJob] = {}
        self._queue: Queue[PipelineJob] = Queue(maxsize=max_queue)
        self._shutdown = threading.Event()
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"pipeline-worker-{idx}", daemon=True)
            for idx in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    # ------------------------------------------------------------------ #
    # Job management
    # ------------------------------------------------------------------ #
    def submit(
        self,
        input_path: Path,
        *,
        episode_id: str,
        show_config_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
        progress_handler: Callable[[PipelineJob], None] | None = None,
    ) -> PipelineJob:
        """Submit a new job to the orchestrator."""
        job_id = uuid.uuid4().hex
        job = PipelineJob(
            job_id=job_id,
            input_path=Path(input_path),
            episode_id=episode_id,
            show_config_path=Path(show_config_path) if show_config_path else None,
            metadata=dict(metadata or {}),
        )
        job._progress_handler = progress_handler
        job.mark(JobStatus.QUEUED, message="Waiting for available worker.")

        try:
            self._queue.put_nowait(job)
        except Full:
            job.mark(JobStatus.FAILED, message="Backpressure: queue is full.")
            raise

        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> PipelineJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[PipelineJob]:
        return list(self._jobs.values())

    def wait(self, job_id: str, timeout: float | None = None) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job found with id {job_id}")
        return job.wait(timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        self._shutdown.set()
        for _ in self._workers:
            self._queue.put_nowait(self._sentinel_job())
        if wait:
            for worker in self._workers:
                worker.join()

    def _sentinel_job(self) -> PipelineJob:
        return PipelineJob(
            job_id="__sentinel__",
            input_path=Path("/dev/null"),
            episode_id="",
            show_config_path=None,
        )

    # ------------------------------------------------------------------ #
    # Worker execution
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            job = self._queue.get()
            if job.job_id == "__sentinel__":
                self._queue.task_done()
                continue
            try:
                self._execute_job(job)
            finally:
                self._queue.task_done()

    def _execute_job(self, job: PipelineJob) -> None:
        job.mark(JobStatus.RUNNING, message="Pipeline started.")
        job.set_progress(0.0)

        try:
            cumulative_weight = 0.0
            for stage in self._stages:
                job.current_stage = stage.name
                progress_adapter = StageProgress(
                    job=job,
                    start_weight=cumulative_weight,
                    stage_weight=stage.weight,
                    total_weight=self._total_weight,
                )
                stage_message = stage.runner(job, self._context, progress_adapter) or ""
                cumulative_weight += stage.weight
                overall = cumulative_weight / self._total_weight
                if not progress_adapter.updated:
                    job.set_progress(overall, message=stage_message)
                else:
                    job.set_progress(overall, message=stage_message)

            job.mark(JobStatus.COMPLETED, message="Pipeline completed successfully.")
            job.set_progress(1.0)
        except Exception as exc:  # pragma: no cover - defensive
            job.error = str(exc)
            job.mark(JobStatus.FAILED, message=f"Pipeline failed: {exc}")
        finally:
            job._completion_event.set()


# ---------------------------------------------------------------------- #
# Default stage implementations
# ---------------------------------------------------------------------- #
def _stage_validate_input(job: PipelineJob, _: PipelineContext, progress: StageProgress) -> str:
    if not job.input_path.exists():
        raise FileNotFoundError(f"Input file '{job.input_path}' does not exist.")
    job.metadata["input_size_bytes"] = job.input_path.stat().st_size
    progress.update(1.0, message="Input validated.")
    return "Input validated."


def _stage_load_show_config(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    if job.show_config_path is None:
        inferred = _infer_show_config_path(job.episode_id, context.paths)
        if inferred is None:
            raise ValueError(
                "show_config path not provided and could not be inferred from episode id."
            )
        job.show_config_path = inferred

    show_config_data = _load_show_config(job.show_config_path)
    show_slug = str(show_config_data.get("show_slug") or job.show_config_path.parent.name)
    display_name = str(show_config_data.get("show_name", show_slug))
    season, episode, variant = _parse_episode_descriptor(job.episode_id)
    descriptor = EpisodeDescriptor(
        show_name=show_slug, season=season, episode=episode, variant=variant
    )

    job.metadata["show_config"] = show_config_data
    job.metadata["show_slug"] = show_slug
    job.metadata["show_display_name"] = display_name
    job.metadata["descriptor"] = descriptor

    progress.update(1.0, message=f"Loaded show config ({display_name})")
    return f"Loaded show config ({display_name})"


def _stage_prepare_directories(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    descriptor = job.metadata.get("descriptor")
    if not isinstance(descriptor, EpisodeDescriptor):
        raise ValueError("Episode descriptor missing; ensure load_show_config stage runs first.")

    override_dir_raw = job.metadata.get("episode_dir_override")
    paths = context.paths
    if override_dir_raw:
        episode_dir = Path(str(override_dir_raw)).expanduser().resolve()
        episode_dir.mkdir(parents=True, exist_ok=True)
    else:
        show_root = paths.show_root(descriptor.show_name)
        show_root.mkdir(parents=True, exist_ok=True)
        episode_dir = paths.episode_directory(descriptor.show_name, descriptor.episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

    job.metadata["episode_dir"] = str(episode_dir)
    progress.update(1.0, message=f"Prepared output directory {episode_dir}")
    return f"Prepared output directory {episode_dir}"


def _stage_extract_audio(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    descriptor = job.metadata.get("descriptor")
    if not isinstance(descriptor, EpisodeDescriptor):
        raise ValueError("Episode descriptor missing; ensure load_show_config stage runs first.")

    extraction_config = AudioExtractionConfig.from_config(context.config)
    extractor = AudioExtractor(
        context.paths,
        extraction_config,
        full_config=dict(context.config),
    )

    input_suffix = job.input_path.suffix.lower()
    if input_suffix == ".wav":
        try:
            clip = audio_io.load_audio(
                job.input_path,
                target_sample_rate=extraction_config.sample_rate,
                mono=extraction_config.channels == 1,
            )
            metadata = extractor._read_audio_metadata(job.input_path)
            extractor._validate_output(metadata, clip)
            report = extractor._run_quality_checks(clip)
        except Exception as exc:  # pragma: no cover - validation guard
            raise AudioExtractionError(f"Failed to validate provided audio: {exc}") from exc

        if not report.passed and extraction_config.thresholds.enforce_strict:
            details = ", ".join(check.details for check in report.checks if not check.passed)
            raise AudioExtractionError(f"Audio quality checks failed: {details}")

        job.metadata["audio_path"] = str(job.input_path)
        job.metadata["audio_metadata"] = metadata
        duration = metadata.get("duration_seconds")
        if isinstance(duration, (int, float)):
            job.metadata["duration_seconds"] = float(duration)

        report_path = job.input_path.parent / "preprocessing_report.json"
        if report_path.exists():
            try:
                job.metadata["preprocessing_report_path"] = str(report_path)
                job.metadata["preprocessing_report"] = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except Exception:  # pragma: no cover - best effort
                LOGGER.debug("Unable to load preprocessing report from %s", report_path)

        progress.update(1.0, message="Audio ready.")
        return f"Using provided audio file {job.input_path}"

    duration_hint = None
    try:
        probe_metadata = extractor.ffmpeg.probe(job.input_path)
        duration_hint = float(probe_metadata.get("format", {}).get("duration", 0.0) or 0.0)
        if duration_hint > 0:
            job.metadata["duration_seconds"] = duration_hint
    except Exception:  # pragma: no cover - best effort
        duration_hint = job.metadata.get("duration_seconds")

    def _progress_cb(update: Any) -> None:
        out_time = getattr(update, "out_time", None)
        if duration_hint and out_time:
            fraction = max(0.0, min(out_time / float(duration_hint), 0.99))
            progress.update(fraction, message=f"Extracting ({out_time:.1f}s)")

    result = extractor.extract(job.input_path, descriptor, progress=_progress_cb)
    job.metadata["audio_path"] = str(result.audio_path)
    job.metadata["audio_metadata"] = result.metadata
    if result.preprocessing_report is not None:
        job.metadata["preprocessing_report"] = result.preprocessing_report
    if result.preprocessing_report_path is not None:
        job.metadata["preprocessing_report_path"] = str(result.preprocessing_report_path)
    progress.update(1.0, message="Audio extracted.")
    return f"Audio saved to {result.audio_path}"


def _stage_transcribe_audio(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    audio_path_str = job.metadata.get("audio_path")
    if not audio_path_str:
        raise ValueError("Audio path missing; ensure extraction stage runs first.")

    transcriber = build_hybrid_transcriber(context.config, context.paths)
    transcription_options: TranscriptionOptions | None = None

    show_config = job.metadata.get("show_config")
    if isinstance(show_config, Mapping):
        prompt = show_config.get("transcription_prompt") or show_config.get("initial_prompt")
        if isinstance(prompt, str) and prompt.strip():
            transcription_options = TranscriptionOptions(initial_prompt=str(prompt).strip())

    start = time.perf_counter()
    transcription = transcriber.transcribe(audio_path_str, options=transcription_options)
    elapsed = time.perf_counter() - start

    job.metadata["transcription"] = transcription
    job.metadata["transcription_seconds"] = elapsed
    progress.update(1.0, message=f"Transcribed {len(transcription.segments)} segments.")
    return f"Transcription produced {len(transcription.segments)} segments."


def _stage_diarize_audio(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    audio_path_str = job.metadata.get("audio_path")
    if not audio_path_str:
        raise ValueError("Audio path missing; ensure extraction stage runs first.")

    diarizer = build_pyannote_diarizer(context.config)
    diarization = diarizer.diarize(audio_path_str)
    job.metadata["diarization"] = diarization
    progress.update(1.0, message=f"Diarized {len(diarization.segments)} segments.")
    return f"Diarization produced {len(diarization.segments)} segments."


def _stage_align_and_export(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:  # noqa: ARG001
    transcription: TranscriptionResult | None = job.metadata.get("transcription")
    diarization: DiarizationResult | None = job.metadata.get("diarization")
    descriptor = job.metadata.get("descriptor")
    episode_dir_str = job.metadata.get("episode_dir")

    if not transcription or not diarization or not isinstance(descriptor, EpisodeDescriptor):
        raise ValueError("Missing transcription/diarization/descriptors for alignment stage.")
    if not episode_dir_str:
        raise ValueError("Episode directory missing; ensure directory stage runs first.")

    alignment_options: dict[str, object] = {}
    alignment_cfg = context.config.get("alignment") if isinstance(context.config, Mapping) else {}
    if isinstance(alignment_cfg, Mapping):
        alignment_options.update(dict(alignment_cfg))

    show_config = job.metadata.get("show_config")
    if isinstance(show_config, Mapping):
        show_alignment = show_config.get("alignment")
        if isinstance(show_alignment, Mapping):
            alignment_options.update(dict(show_alignment))

    name_corrector = None
    features_cfg = context.config.get("features") if isinstance(context.config, Mapping) else {}
    auto_correct_names = True
    if isinstance(features_cfg, Mapping):
        auto_correct_names = bool(features_cfg.get("auto_correct_names", True))
    if isinstance(show_config, Mapping):
        auto_correct_names = bool(show_config.get("auto_correct_names", auto_correct_names))
        try:
            name_corrector = NameCorrector(show_config)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Name corrector unavailable: %s", exc)
            name_corrector = None

    builder = TranscriptBuilder(name_corrector=name_corrector)
    pipeline = TranscriptPipeline(builder=builder, alignment_options=alignment_options)
    progress.update(0.2, message="Aligning ASR and diarization outputs.")
    metadata = {
        "episode_id": descriptor.episode_id,
        "show": job.metadata.get("show_display_name", descriptor.show_name),
    }
    transcript_result = pipeline.run(
        transcription,
        diarization,
        episode_metadata=metadata,
        auto_correct_names=auto_correct_names,
    )

    episode_dir = Path(episode_dir_str)
    episode_dir.mkdir(parents=True, exist_ok=True)

    progress.update(0.6, message="Writing transcript exports.")
    transcript_txt = episode_dir / "transcript_final.txt"
    transcript_txt.write_text(render_plain_text(transcript_result.document), encoding="utf-8")

    transcript_srt = episode_dir / "transcript_final.srt"
    transcript_srt.write_text(render_srt(transcript_result.document), encoding="utf-8")

    payload = build_transcript_payload(
        transcript_result.document,
        alignment=transcript_result.alignment,
        corrections=transcript_result.corrections,
    )

    job.metadata["alignment"] = transcript_result.alignment
    job.metadata["transcript_document"] = transcript_result.document
    job.metadata["name_corrections"] = transcript_result.corrections

    transcript_json = episode_dir / "transcript_final.json"

    existing_created: str | None = None
    existing_updated: str | None = None
    if transcript_json.exists():
        try:
            existing_payload = json.loads(transcript_json.read_text(encoding="utf-8"))
            timestamps_block = existing_payload.get("metadata", {}).get("timestamps", {})
            existing_created = timestamps_block.get("created")
            existing_updated = timestamps_block.get("updated")
        except Exception:  # pragma: no cover - best effort to preserve history
            existing_created = None
            existing_updated = None

    metadata_block = payload.setdefault("metadata", {})
    timestamps_block = metadata_block.setdefault("timestamps", {})
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    timestamps_block["updated"] = now
    if existing_created:
        timestamps_block["created"] = existing_created
    else:
        timestamps_block.setdefault("created", now)
    if existing_updated and existing_updated != timestamps_block["updated"]:
        timestamps_block.setdefault("previous_updated", existing_updated)

    transcript_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    job.metadata["transcript_paths"] = {
        "txt": str(transcript_txt),
        "srt": str(transcript_srt),
        "json": str(transcript_json),
    }

    progress.update(1.0, message="Transcript exports written.")
    return "Transcript exports completed."


def _stage_identify_speakers(
    job: PipelineJob, context: PipelineContext, progress: StageProgress
) -> str:
    features_cfg = context.config.get("features") if isinstance(context.config, Mapping) else {}
    if not isinstance(features_cfg, Mapping) or not bool(
        features_cfg.get("enable_voice_bank", False)
    ):
        progress.update(1.0, message="Voice bank disabled; skipping.")
        return "Speaker identification skipped (feature disabled)."

    descriptor = job.metadata.get("descriptor")
    audio_path_str = job.metadata.get("audio_path")
    diarization: DiarizationResult | None = job.metadata.get("diarization")
    alignment = job.metadata.get("alignment")

    if (
        not isinstance(descriptor, EpisodeDescriptor)
        or not audio_path_str
        or diarization is None
        or alignment is None
    ):
        progress.update(1.0, message="Speaker identification prerequisites missing.")
        return "Speaker identification skipped (missing prerequisites)."

    try:
        voice_pipeline = build_voice_bank_pipeline(context.config, context.paths)
    except SpeakerIdentificationError as exc:
        LOGGER.warning("Speaker identification unavailable: %s", exc)
        progress.update(1.0, message="Speaker identification unavailable.")
        return "Speaker identification skipped (dependency missing)."

    progress.update(0.2, message="Computing speaker embeddings.")
    result = voice_pipeline.identify(
        episode_id=descriptor.episode_id,
        audio_path=Path(audio_path_str),
        diarization=diarization,
        alignment=alignment,
    )

    result_dict = result.to_dict()
    job.metadata["voice_identification"] = result_dict

    assignments_by_cluster = {
        assignment["cluster_id"]: assignment
        for assignment in result_dict.get("assignments", [])
        if assignment.get("matched")
    }

    transcript_doc = job.metadata.get("transcript_document")
    transcript_paths = job.metadata.get("transcript_paths")
    corrections = job.metadata.get("name_corrections")

    updated_transcript = False

    if assignments_by_cluster and isinstance(transcript_doc, TranscriptDocument):
        for segment in transcript_doc.segments:
            metadata = segment.metadata
            original = metadata.get("original_speaker") or segment.speaker
            assignment = assignments_by_cluster.get(original)
            if not assignment:
                continue
            target_name = assignment.get("display_name") or assignment.get("speaker_key")
            if not target_name or segment.speaker == target_name:
                continue
            metadata.setdefault("original_speaker", original)
            metadata["voice_bank_match"] = {
                key: assignment[key]
                for key in ("speaker_key", "display_name", "similarity", "auto_registered")
                if assignment.get(key) is not None
            }
            segment.speaker = target_name
            if assignment.get("similarity") is not None:
                try:
                    segment.speaker_confidence = float(assignment["similarity"])  # type: ignore[assignment]
                except (TypeError, ValueError):
                    pass
            updated_transcript = True

    if assignments_by_cluster and isinstance(alignment, AlignmentResult):
        for aligned_segment in alignment.segments:
            original = aligned_segment.metadata.get("original_speaker") or aligned_segment.speaker
            assignment = assignments_by_cluster.get(original)
            if not assignment:
                continue
            target_name = assignment.get("display_name") or assignment.get("speaker_key")
            if target_name and aligned_segment.speaker != target_name:
                aligned_segment.metadata.setdefault("original_speaker", original)
                aligned_segment.speaker = target_name
                updated_transcript = True

        alignment.metadata.speakers = [
            assignments_by_cluster.get(name, {}).get("display_name")
            or assignments_by_cluster.get(name, {}).get("speaker_key")
            or name
            for name in alignment.metadata.speakers
        ]

    if updated_transcript and isinstance(transcript_paths, Mapping):
        txt_path_raw = transcript_paths.get("txt")
        srt_path_raw = transcript_paths.get("srt")
        json_path_raw = transcript_paths.get("json")

        if txt_path_raw and srt_path_raw and json_path_raw:
            txt_path = Path(txt_path_raw)
            srt_path = Path(srt_path_raw)
            json_path = Path(json_path_raw)

            txt_path.write_text(render_plain_text(transcript_doc), encoding="utf-8")
            srt_path.write_text(render_srt(transcript_doc), encoding="utf-8")

            extras = {"voice_identification": result_dict}
            payload = build_transcript_payload(
                transcript_doc,
                alignment=alignment,
                corrections=corrections if isinstance(corrections, list) else None,
                extras=extras,
            )
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assigned = result.matched
    progress.update(1.0, message=f"Matched {assigned} speakers.")
    return f"Matched {assigned} speakers to voice bank."


DEFAULT_STAGES = [
    StageDefinition("Validate Input", _stage_validate_input, weight=0.05),
    StageDefinition("Load Show Config", _stage_load_show_config, weight=0.10),
    StageDefinition("Prepare Directories", _stage_prepare_directories, weight=0.10),
    StageDefinition("Extract Audio", _stage_extract_audio, weight=0.25),
    StageDefinition("Transcribe Audio", _stage_transcribe_audio, weight=0.25),
    StageDefinition("Diarize Audio", _stage_diarize_audio, weight=0.15),
    StageDefinition("Align & Export Transcript", _stage_align_and_export, weight=0.05),
    StageDefinition("Identify Speakers", _stage_identify_speakers, weight=0.05),
]


# ---------------------------------------------------------------------- #
# Convenience helpers
# ---------------------------------------------------------------------- #
def build_default_context(
    env: str = "dev", overrides: Mapping[str, Any] | None = None
) -> PipelineContext:
    config = load_config(env, overrides=overrides)
    paths = build_paths(config)
    paths.ensure_directories()
    return PipelineContext(config=config, paths=paths, environment=env)
