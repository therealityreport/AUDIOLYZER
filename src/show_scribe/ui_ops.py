"""Thin wrappers used by the Streamlit UI to orchestrate pipeline actions."""

from __future__ import annotations

# Import exceptions from dedicated module to avoid circular imports
from .exceptions import AudioPreprocessingCancelled, AudioPreprocessorCancelled

__all__ = [
    "AudioPreprocessingCancelled",
    "AudioPreprocessorCancelled",
    "raise_if_cancelled",
    "AudioPreprocessingError",
    "extract_and_enhance",
    "request_audio_cancellation",
    "reset_audio_cancellation",
    "cancel_active_audio_runs",
    "create_audio",
    "ensure_episode_scaffold",
    "read_metadata",
    "resolve_artifact_path",
    "update_metadata",
    "run_transcription",
    "run_dual_transcription",
    "run_diarization",
    "run_bleep_detection",
    "run_speaker_id",
    "compute_analytics",
    "get_stage_status",
]


def raise_if_cancelled(cancelled: bool, msg: str = "Preprocessing cancelled by user.") -> None:
    """Raise ``AudioPreprocessingCancelled`` when a cancellation flag is set."""

    if cancelled:
        raise AudioPreprocessingCancelled(msg)


import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pipeline import audio_ops as _audio_ops

AudioPreprocessingError = _audio_ops.AudioPreprocessingError
extract_and_enhance = _audio_ops.extract_and_enhance

if hasattr(_audio_ops, "AudioPreprocessingCancelled"):

    class _PipelineAudioCancelled(
        _audio_ops.AudioPreprocessingCancelled, AudioPreprocessingCancelled
    ):  # type: ignore[misc]
        """Adapter ensuring pipeline cancellations satisfy the UI contract."""

        pass

    AudioPreprocessingCancelled = _PipelineAudioCancelled
    AudioPreprocessorCancelled = AudioPreprocessingCancelled

try:
    request_audio_cancellation = _audio_ops.request_audio_cancellation  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - legacy compatibility

    def request_audio_cancellation() -> bool:
        """Fallback cancellation request when pipeline does not expose cancellation."""

        return False


try:
    reset_audio_cancellation = _audio_ops.reset_audio_cancellation  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - legacy compatibility

    def reset_audio_cancellation() -> None:
        """Fallback cancellation reset when pipeline does not expose cancellation."""

        return None


from show_scribe.pipelines.alignment.align_asr_diar import (
    AlignedSegment,
    AlignedWord,
    AlignmentMetadata,
    AlignmentResult,
)
from show_scribe.pipelines.audio_preprocessing import select_transcription_inputs
from show_scribe.pipelines.asr import (
    LocalWhisperTranscriber,
    TranscriptionOptions,
    build_hybrid_transcriber,
)
from show_scribe.pipelines.asr.whisper_local import (
    SegmentTranscription,
    TranscriptionMetadata,
    TranscriptionResult,
    WordTiming,
)
from show_scribe.pipelines.diarization.pyannote_pipeline import (
    DiarizationMetadata,
    DiarizationResult,
    DiarizationSegment,
    build_pyannote_diarizer,
)
from show_scribe.pipelines.orchestrator import build_default_context
from show_scribe.pipelines.speaker_id.voice_bank import (
    SpeakerIdentificationError,
    build_voice_bank_pipeline,
)
from show_scribe.pipelines.transcript.builder import (
    TranscriptBuilder,
    TranscriptDocument,
    TranscriptSegment,
)
from show_scribe.pipelines.transcript.export_json import build_transcript_payload
from show_scribe.pipelines.transcript.export_srt import render_srt
from show_scribe.pipelines.transcript.export_text import render_plain_text
from show_scribe.pipelines.transcript.pipeline import TranscriptPipeline
from show_scribe.utils.logging import get_logger
from show_scribe.utils.name_correction import NameCorrector
from show_scribe.utils.audio_io import (
    AudioClip,
    extract_segment as _extract_segment,
    load_audio as _load_audio,
    save_audio as _save_audio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHOWS_ROOT = PROJECT_ROOT / "data" / "shows"
DEFAULT_AUDIO_LIBRARY_ROOT = PROJECT_ROOT / "data" / "AUDIO FILES"
METADATA_FILENAME = "metadata.json"

ARTIFACT_DEFAULTS = {
    # Audio artifacts
    "audio_extracted": "audio/audio_extracted.wav",  # Raw audio extracted from video
    "audio_raw": "audio/audio_extracted.wav",  # Alias for convenience in UI
    "audio_vocals": "audio/audio_vocals.wav",  # Separated vocals (no background)
    "audio_enhanced_vocals": "audio/audio_enhanced_vocals.wav",  # Enhanced vocals (no background)
    "audio_enhanced_mix": "audio/audio_enhanced_mix.wav",  # Enhanced full mix (background preserved)
    # Back-compat keys used by earlier pages/pipelines
    "audio_enhanced": "audio/audio_enhanced_vocals.wav",
    "audio_processed": "audio/audio_enhanced_vocals.wav",
    # Reports and downstream artifacts
    "preprocessing_report": "audio/reports/preprocessing_report.json",
    # Dual transcription workflow
    "transcript_raw_vocals": "transcripts/transcript_raw.vocals.json",
    "transcript_raw_mix": "transcripts/transcript_raw.mix.json",
    "aligned_vocals": "transcripts/aligned_vocals.jsonl",
    "aligned_mix": "transcripts/aligned_mix.jsonl",
    "transcript_draft_txt": "transcripts/transcript_draft.txt",
    "transcript_draft_srt": "transcripts/transcript_draft.srt",
    # Legacy single-path workflow
    "transcript_raw": "transcripts/transcript_raw.json",
    "diarization": "transcripts/diarization.json",
    "transcript_final": "transcripts/transcript_final.json",
    "bleeps": "transcripts/bleeps.json",
    "analytics": "analytics/analytics.json",
}

ARTIFACT_GLOBS = {
    "audio_extracted": ["audio/*_audio_extracted.*", "*_audio_extracted.*", "audio_extracted.*"],
    "audio_raw": ["audio/*_audio_extracted.*", "*_audio_extracted.*", "audio_extracted.*"],
    "audio_vocals": ["audio/*_audio_vocals.*", "*_audio_vocals.*", "audio_vocals.*"],
    "audio_enhanced_vocals": [
        "audio/*_audio_enhanced_vocals.*",
        "*_audio_enhanced_vocals.*",
        "audio_enhanced_vocals.*",
    ],
    "audio_enhanced_mix": [
        "audio/*_audio_enhanced_mix.*",
        "*_audio_enhanced_mix.*",
        "audio_enhanced_mix.*",
    ],
    # Back-compat fallbacks
    "audio_enhanced": [
        "audio/*_audio_enhanced_vocals.*",
        "*_audio_enhanced_vocals.*",
        "audio_enhanced_vocals.*",
        # legacy name
        "audio/*_audio_enhanced.*",
        "*_audio_enhanced.*",
        "audio_enhanced.*",
    ],
    "audio_processed": [
        "audio/*_audio_enhanced_vocals.*",
        "*_audio_enhanced_vocals.*",
        "audio_enhanced_vocals.*",
        # legacy name
        "audio/*_audio_enhanced.*",
        "*_audio_enhanced.*",
        "audio_enhanced.*",
    ],
    "preprocessing_report": ["audio/reports/*_preprocessing_report.json"],
    # Dual transcription workflow
    "transcript_raw_vocals": [
        "transcripts/*_transcript_raw.vocals.json",
        "transcripts/transcript_raw.vocals.json",
    ],
    "transcript_raw_mix": [
        "transcripts/*_transcript_raw.mix.json",
        "transcripts/transcript_raw.mix.json",
    ],
    "aligned_vocals": ["transcripts/*_aligned_vocals.jsonl", "transcripts/aligned_vocals.jsonl"],
    "aligned_mix": ["transcripts/*_aligned_mix.jsonl", "transcripts/aligned_mix.jsonl"],
    "transcript_draft_txt": ["transcripts/transcript_draft.txt"],
    "transcript_draft_srt": ["transcripts/transcript_draft.srt"],
    # Legacy single-path workflow
    "transcript_raw": ["transcripts/*_transcript_raw.json"],
    "diarization": ["transcripts/*_diarization.json"],
    "transcript_final": ["transcripts/*_transcript_final.json"],
    "bleeps": ["transcripts/*bleeps*"],
    "analytics": ["analytics/*.json"],
}

LOGGER = get_logger(__name__)

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SRT_TIMESTAMP_RE = re.compile(r"(\d+:\d{2}:\d{2}),(\d{3})")

ProgressReporter = Callable[[str, float | None], None]


def _timestamp() -> str:
    value = datetime.now(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_shows_root() -> Path:
    try:
        from ui.streamlit_app.components.settings_manager import SettingsManager  # type: ignore
    except Exception:
        raw = os.getenv("SHOWS_ROOT")
        return Path(raw).expanduser().resolve() if raw else DEFAULT_SHOWS_ROOT.resolve()

    try:
        return SettingsManager.get_shows_root()
    except Exception:
        raw = os.getenv("SHOWS_ROOT")
        return Path(raw).expanduser().resolve() if raw else DEFAULT_SHOWS_ROOT.resolve()


def _current_environment() -> str:
    return os.getenv("SHOW_SCRIBE_ENVIRONMENT", "dev")


def _resolve_audio_library_root() -> Path | None:
    raw = os.getenv("AUDIO_LIBRARY_ROOT")
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.append(DEFAULT_AUDIO_LIBRARY_ROOT)
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        try:
            return candidate.resolve()
        except OSError:
            return candidate.expanduser()
    return None


def _resolve_existing_path(episode_dir: Path, candidate: str | Path | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    search_order = []
    if path.is_absolute():
        search_order.append(path)
    else:
        search_order.extend(
            [
                episode_dir / path,
                episode_dir.parent / path,
                episode_dir.parent.parent / path,
                PROJECT_ROOT / path,
            ]
        )
        search_order.append(path)
    for option in search_order:
        try_path = option.expanduser()
        if try_path.exists():
            return try_path.resolve()
    return None


def _ensure_episode_audio_path(episode_dir: Path, metadata: Mapping[str, Any]) -> Path:
    try:
        vocals_path, _ = select_transcription_inputs(episode_dir)
        return vocals_path
    except FileNotFoundError:
        pass

    artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else {}
    preferred: list[str | Path | None] = []
    fallback_raw: list[str | Path | None] = []
    if isinstance(artifacts, Mapping):
        preferred.extend(
            [
                artifacts.get("audio_enhanced_vocals") or artifacts.get("audio_enhanced"),
                artifacts.get("audio_vocals"),
                artifacts.get("audio_enhanced_mix"),
                artifacts.get("audio_processed"),
            ]
        )
        fallback_raw.extend(
            [
                artifacts.get("audio_extracted"),
                artifacts.get("audio_raw"),
            ]
        )
    preferred.extend(
        [
            "audio/audio_enhanced_vocals.wav",
            "audio/audio_vocals.wav",
            "audio/audio_enhanced_mix.wav",
            "audio/audio_processed.wav",
            "audio_enhanced_vocals.wav",
            "audio_vocals.wav",
            "audio_enhanced_mix.wav",
            "audio_processed.wav",
        ]
    )
    fallback_raw.extend(
        [
            "audio/audio_enhanced_vocals.wav",
            "audio/audio_vocals.wav",
            "audio/audio_extracted.wav",
            "audio/audio_processed.wav",
            "audio_enhanced_vocals.wav",
            "audio_vocals.wav",
            "audio_extracted.wav",
            "audio_processed.wav",
        ]
    )
    for candidate in preferred:
        resolved = _resolve_existing_path(episode_dir, candidate)
        if resolved:
            return resolved
    raw_hits = [
        resolved
        for candidate in fallback_raw
        if (resolved := _resolve_existing_path(episode_dir, candidate)) is not None
    ]
    if raw_hits:
        paths = "\n".join(str(path) for path in raw_hits)
        raise FileNotFoundError(
            "Enhanced or vocal-isolated audio artifacts were not found. "
            "Only raw extracted audio exists:\n"
            f"{paths}\n"
            "Re-run CREATE AUDIO to generate enhanced vocals before continuing."
        )
    raise FileNotFoundError(
        f"No enhanced or vocal-isolated audio artifacts found in {episode_dir}. "
        "Run CREATE AUDIO before transcription or diarization."
    )


def _resolve_show_config_path(episode_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()

    candidates = [
        episode_dir / path,
        episode_dir.parent.parent / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded.resolve()
    return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_artifact_prefix(show_slug: str, episode_id: str, created_at: str | None) -> str:
    sanitized_slug = _SANITIZE_RE.sub("_", show_slug or "").strip("_") or "show"
    sanitized_episode = _SANITIZE_RE.sub("_", episode_id or "").strip("_") or "episode"
    created_dt = _parse_timestamp(created_at) or datetime.now(timezone.utc)
    date_tag = created_dt.strftime("%Y%m%d")
    prefix = f"{sanitized_slug}_{sanitized_episode}_{date_tag}"
    return _SANITIZE_RE.sub("_", prefix).strip("_") or f"{sanitized_slug}_{date_tag}"


def _relative_to_episode(episode_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(episode_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_speakers(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if isinstance(speaker, str) and speaker and speaker not in seen:
            seen.add(speaker)
            ordered.append(speaker)
    return ordered


def _transcription_from_payload(payload: Mapping[str, Any]) -> TranscriptionResult:
    segment_payloads = payload.get("segments") or []
    segments: list[SegmentTranscription] = []
    for index, entry in enumerate(segment_payloads):
        if not isinstance(entry, Mapping):
            continue
        words_payload = entry.get("words") or []
        words: list[WordTiming] = []
        for word_entry in words_payload:
            if not isinstance(word_entry, Mapping):
                continue
            word_text = str(word_entry.get("word", "")).strip()
            if not word_text:
                continue
            words.append(
                WordTiming(
                    word=word_text,
                    start=_to_float(word_entry.get("start")) or 0.0,
                    end=_to_float(word_entry.get("end")) or 0.0,
                    probability=_to_float(word_entry.get("probability")) or 0.0,
                )
            )
        segments.append(
            SegmentTranscription(
                segment_id=_to_int(entry.get("id")) or index,
                start=_to_float(entry.get("start")) or 0.0,
                end=_to_float(entry.get("end")) or 0.0,
                text=str(entry.get("text") or ""),
                confidence=_to_float(entry.get("confidence")) or 0.0,
                words=words,
            )
        )

    metadata_payload = payload.get("metadata") or {}
    transcription_metadata = TranscriptionMetadata(
        language=str(metadata_payload.get("language") or None) or None,
        duration=_to_float(metadata_payload.get("duration")),
        detected_language_probability=_to_float(metadata_payload.get("language_probability")),
        cost_usd=_to_float(metadata_payload.get("cost_usd")),
    )
    return TranscriptionResult(segments=segments, metadata=transcription_metadata)


def _alignment_from_payload(payload: Mapping[str, Any]) -> AlignmentResult:
    segment_payloads = payload.get("segments") or []
    segments: list[AlignedSegment] = []
    for index, entry in enumerate(segment_payloads):
        if not isinstance(entry, Mapping):
            continue
        words_payload = entry.get("words") or []
        words: list[AlignedWord] = []
        for word_entry in words_payload:
            if not isinstance(word_entry, Mapping):
                continue
            word_text = str(word_entry.get("word", "")).strip()
            if not word_text:
                continue
            words.append(
                AlignedWord(
                    word=word_text,
                    start=_to_float(word_entry.get("start")) or 0.0,
                    end=_to_float(word_entry.get("end")) or 0.0,
                    speaker=str(word_entry.get("speaker") or entry.get("speaker") or ""),
                    probability=_to_float(word_entry.get("probability")),
                )
            )
        segments.append(
            AlignedSegment(
                segment_id=_to_int(entry.get("id")) or index,
                start=_to_float(entry.get("start")) or 0.0,
                end=_to_float(entry.get("end")) or 0.0,
                speaker=str(entry.get("speaker") or ""),
                text=str(entry.get("text") or ""),
                words=words,
                speaker_confidence=_to_float(entry.get("speaker_confidence")),
                metadata=dict(entry.get("metadata") or {}),
            )
        )

    metadata_payload = payload.get("metadata") or {}
    speakers_raw = metadata_payload.get("speakers")
    if isinstance(speakers_raw, Sequence):
        speakers = [str(item) for item in speakers_raw if str(item).strip()]
    else:
        speakers = []
    metadata = AlignmentMetadata(
        asr_segment_count=_to_int(metadata_payload.get("asr_segment_count")) or len(segments),
        diarization_segment_count=_to_int(metadata_payload.get("diarization_segment_count")) or 0,
        words_aligned_count=_to_int(metadata_payload.get("words_aligned_count")) or 0,
        unaligned_word_count=_to_int(metadata_payload.get("unaligned_word_count")) or 0,
        speakers=speakers,
    )
    return AlignmentResult(segments=segments, metadata=metadata)


def _diarization_from_payload(payload: Mapping[str, Any]) -> DiarizationResult:
    segment_payloads = payload.get("segments") or []
    segments: list[DiarizationSegment] = []
    for index, entry in enumerate(segment_payloads):
        if not isinstance(entry, Mapping):
            continue
        segments.append(
            DiarizationSegment(
                segment_id=_to_int(entry.get("id")) or index,
                start=_to_float(entry.get("start")) or 0.0,
                end=_to_float(entry.get("end")) or 0.0,
                speaker=str(entry.get("speaker") or f"cluster_{index}"),
                track=entry.get("track"),
                confidence=_to_float(entry.get("confidence")),
            )
        )

    metadata_payload = payload.get("metadata") or {}
    diar_metadata = DiarizationMetadata(
        model=str(metadata_payload.get("model") or ""),
        speaker_count=_to_int(metadata_payload.get("speaker_count")) or len(segments),
        duration=_to_float(metadata_payload.get("duration")),
        inference_seconds=_to_float(metadata_payload.get("inference_seconds")),
        parameters=dict(metadata_payload.get("parameters") or {}),
    )
    return DiarizationResult(segments=segments, metadata=diar_metadata)


def resolve_artifact_path(episode_dir: Path, artifact_key: str) -> Path | None:
    """Return the resolved path for a known artifact key, if it exists."""

    metadata = read_metadata(episode_dir) or {}
    artifacts = metadata.get("artifacts")
    candidate = None
    if isinstance(artifacts, Mapping):
        candidate = artifacts.get(artifact_key)
    resolved = _resolve_existing_path(episode_dir, candidate)
    if resolved:
        return resolved
    shared_artifacts = metadata.get("shared_artifacts")
    if isinstance(shared_artifacts, Mapping):
        fallback_candidate = shared_artifacts.get(artifact_key)
        resolved_local = _resolve_existing_path(episode_dir, fallback_candidate)
        if resolved_local:
            return resolved_local
    default_relative = ARTIFACT_DEFAULTS.get(artifact_key)
    if default_relative:
        fallback = episode_dir / default_relative
        if fallback.exists():
            return fallback.resolve()
    patterns = ARTIFACT_GLOBS.get(artifact_key, [])
    for pattern in patterns:
        for match in episode_dir.glob(pattern):
            if match.exists():
                return match.resolve()
    return None


def _episode_metadata_path(episode_dir: Path) -> Path:
    return episode_dir / METADATA_FILENAME


def read_metadata(episode_dir: Path) -> dict[str, Any] | None:
    """Return parsed metadata for an episode if available."""

    metadata_path = _episode_metadata_path(episode_dir)
    if not metadata_path.exists():
        return None
    try:
        raw = metadata_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_metadata(episode_dir: Path, data: dict[str, Any]) -> None:
    """Persist metadata to disk, ensuring directory structure exists."""

    episode_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = _episode_metadata_path(episode_dir)
    payload = dict(data)
    payload.setdefault("updated_at", _timestamp())
    temp_path = metadata_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp_path.replace(metadata_path)


def update_metadata(episode_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into existing metadata, returning the merged payload."""

    current = read_metadata(episode_dir) or {}
    merged = dict(current)

    artifacts_update = updates.get("artifacts")
    if isinstance(artifacts_update, dict):
        base = dict(merged.get("artifacts") or {})
        for key, value in artifacts_update.items():
            if value is None:
                base.pop(key, None)
            else:
                base[key] = value
        merged["artifacts"] = base

    shared_update = updates.get("shared_artifacts")
    if isinstance(shared_update, dict):
        base_shared = dict(merged.get("shared_artifacts") or {})
        for key, value in shared_update.items():
            if value is None:
                base_shared.pop(key, None)
            else:
                base_shared[key] = value
        if base_shared:
            merged["shared_artifacts"] = base_shared
        else:
            merged.pop("shared_artifacts", None)

    for key, value in updates.items():
        if key == "artifacts":
            continue
        if key == "shared_artifacts":
            continue
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value

    merged.setdefault("created_at", current.get("created_at", _timestamp()))
    show_slug = str(merged.get("show_slug") or current.get("show_slug") or "")
    episode_id = str(merged.get("episode_id") or current.get("episode_id") or "")
    merged["artifact_prefix"] = _build_artifact_prefix(
        show_slug, episode_id, merged.get("created_at")
    )
    merged["updated_at"] = _timestamp()
    write_metadata(episode_dir, merged)
    return merged


def ensure_episode_scaffold(
    show_slug: str,
    episode_id: str,
    video_path: Path,
    preset_path: Path,
) -> Path:
    """Create canonical directory layout and metadata for an episode."""

    source_video = Path(video_path).expanduser().resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    preset_path = Path(preset_path).expanduser()
    shows_root = _resolve_shows_root()
    episode_dir = shows_root / show_slug / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    dest_suffix = ".mp4" if source_video.suffix.lower() == ".mp4" else source_video.suffix or ".mp4"
    dest_video = episode_dir / f"{episode_id}{dest_suffix}"
    if not dest_video.exists():
        try:
            dest_video.symlink_to(source_video)
        except OSError:
            shutil.copy2(source_video, dest_video)

    metadata = read_metadata(episode_dir) or {}
    metadata.setdefault("created_at", _timestamp())
    metadata.update(
        {
            "episode_id": episode_id,
            "show_slug": show_slug,
            "source_video_path": str(source_video),
            "preset": str(preset_path),
            "updated_at": _timestamp(),
        }
    )
    metadata.setdefault("artifacts", {})
    metadata.setdefault("created_at", _timestamp())
    metadata["artifact_prefix"] = _build_artifact_prefix(
        show_slug,
        episode_id,
        metadata.get("created_at"),
    )
    (episode_dir / "audio").mkdir(parents=True, exist_ok=True)
    (episode_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (episode_dir / "analytics").mkdir(parents=True, exist_ok=True)
    write_metadata(episode_dir, metadata)
    return episode_dir


def cancel_active_audio_runs() -> bool:
    """Signal running audio preprocessing commands to terminate."""

    return request_audio_cancellation()


def create_audio(
    episode_dir: Path,
    video_path: Path,
    preset_path: Path,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Path | None]:
    """Execute the audio extraction and enhancement flow for an episode."""

    reset_audio_cancellation()

    episode_dir = episode_dir.expanduser().resolve()
    preset_path = preset_path.expanduser().resolve()
    video_path = video_path.expanduser().resolve()

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    _emit("Starting audio extraction and preprocessing…", 0.20)
    failure: AudioPreprocessingError | None = None
    artifacts: dict[str, Path | None]
    try:
        artifacts = extract_and_enhance(video_path, episode_dir, preset_path)
        _emit("Extraction and preprocessing complete; organising artifacts…", 0.4)
    except AudioPreprocessingError as exc:
        _emit("Audio preprocessing failed; collecting any partial artifacts…", 0.35)
        failure = exc
        artifacts = dict(getattr(exc, "artifacts", {}) or {})
    except Exception:
        raise

    metadata = read_metadata(episode_dir) or {}
    metadata_show_slug = str(metadata.get("show_slug") or "").strip()
    if not metadata_show_slug:
        try:
            metadata_show_slug = episode_dir.parents[1].name
        except IndexError:
            metadata_show_slug = episode_dir.parent.name
    metadata_episode_id = str(metadata.get("episode_id") or "").strip() or episode_dir.name

    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        metadata_show_slug,
        metadata_episode_id,
        metadata.get("created_at"),
    )

    audio_dir = episode_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def _first_existing(*candidates: Path | str | None) -> Path | None:
        for candidate in candidates:
            if not candidate:
                continue
            try:
                path = Path(candidate).expanduser()
            except Exception:
                continue
            if path.exists():
                return path
        return None

    def _promote_to_audio(source: Path | None, filename: str) -> Path | None:
        if not source:
            return None
        try:
            source_path = Path(source).resolve()
        except Exception:
            return None
        if not source_path.exists():
            return None
        target = audio_dir / f"{prefix}_{filename}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            if source_path == target.resolve():
                return target
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(source_path), str(target))
        return target

    raw_sources = {
        "extracted": _first_existing(
            artifacts.get("extracted"),
            episode_dir / "audio_extracted.wav",
            audio_dir / f"{prefix}_audio_extracted.wav",
        ),
        "vocals": _first_existing(
            artifacts.get("vocals"),
            episode_dir / "audio_vocals.wav",
            audio_dir / f"{prefix}_audio_vocals.wav",
        ),
        "enhanced_vocals": _first_existing(
            artifacts.get("enhanced_vocals") or artifacts.get("enhanced"),
            episode_dir / "audio_enhanced_vocals.wav",
            episode_dir / "audio_enhanced.wav",  # legacy
            audio_dir / f"{prefix}_audio_enhanced_vocals.wav",
            audio_dir / f"{prefix}_audio_enhanced.wav",  # legacy
        ),
        "enhanced_mix": _first_existing(
            artifacts.get("enhanced_mix"),
            episode_dir / "audio_enhanced_mix.wav",
            audio_dir / f"{prefix}_audio_enhanced_mix.wav",
        ),
        "report": _first_existing(
            artifacts.get("report"),
            audio_dir / "reports" / f"{prefix}_preprocessing_report.json",
        ),
    }

    renamed: dict[str, Path | None] = {}

    artifact_sequence = [
        ("extracted", "audio_extracted.wav", raw_sources.get("extracted")),
        ("vocals", "audio_vocals.wav", raw_sources.get("vocals")),
        ("enhanced_vocals", "audio_enhanced_vocals.wav", raw_sources.get("enhanced_vocals")),
        ("enhanced_mix", "audio_enhanced_mix.wav", raw_sources.get("enhanced_mix")),
    ]
    total_artifacts = len(artifact_sequence) + 1  # +1 for the report
    base_progress = 0.30
    progress_span = 0.55

    def _artifact_progress(step_index: int) -> float:
        if total_artifacts <= 0:
            return 0.9
        clamped = max(0, min(step_index, total_artifacts))
        return base_progress + progress_span * (clamped / total_artifacts)

    for index, (key, filename, source_option) in enumerate(artifact_sequence, start=1):
        promoted = _promote_to_audio(source_option, filename)
        renamed[key] = promoted
        if promoted:
            _emit(
                f"[{index}/{total_artifacts}] {filename} saved ({promoted.name}).",
                _artifact_progress(index),
            )
        else:
            _emit(
                f"[{index}/{total_artifacts}] {filename} not produced; skipping.",
                _artifact_progress(index),
            )

    # Guard against edge cases where earlier moves failed – sweep legacy root files now.
    legacy_candidates: dict[str, tuple[Path, str]] = {
        "extracted": (episode_dir / "audio_extracted.wav", "audio_extracted.wav"),
        "vocals": (episode_dir / "audio_vocals.wav", "audio_vocals.wav"),
        # Support both new and legacy filenames for enhanced vocals
        "enhanced_vocals": (episode_dir / "audio_enhanced_vocals.wav", "audio_enhanced_vocals.wav"),
        "enhanced": (episode_dir / "audio_enhanced.wav", "audio_enhanced_vocals.wav"),
        # Enhanced mix
        "enhanced_mix": (episode_dir / "audio_enhanced_mix.wav", "audio_enhanced_mix.wav"),
    }
    for key, (legacy_path, filename) in legacy_candidates.items():
        if not legacy_path.exists():
            continue
        target = audio_dir / f"{prefix}_{filename}"
        if target.exists() and target.resolve() == legacy_path.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            target.unlink()
        shutil.move(str(legacy_path), str(target))
        renamed[key] = target
        _emit(f"Relocated legacy {filename} to {target.name}", 0.78)

    def _ensure_alias(source: Path | None, destination: Path) -> Path | None:
        if not source:
            with contextlib.suppress(OSError):
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
            return None
        try:
            source_path = Path(source).expanduser().resolve()
        except Exception:
            return None
        if not source_path.exists():
            with contextlib.suppress(OSError):
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists() or destination.is_symlink():
                if destination.samefile(source_path):
                    return destination
                destination.unlink()
        except OSError:
            with contextlib.suppress(OSError):
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
        try:
            shutil.copy2(str(source_path), str(destination))
        except shutil.SameFileError:
            pass
        return destination

    canonical_aliases: dict[str, Path | None] = {}
    alias_sources = {
        "audio_extracted.wav": renamed.get("extracted"),
        "audio_vocals.wav": renamed.get("vocals"),
        "audio_enhanced.wav": renamed.get("enhanced"),
        "audio_processed.wav": renamed.get("enhanced"),
    }
    for filename, source_path in alias_sources.items():
        alias_path = episode_dir / filename
        canonical_aliases[filename] = _ensure_alias(source_path, alias_path)

    library_root = _resolve_audio_library_root()
    library_dir: Path | None = None
    shared_artifacts: dict[str, str] = {}
    if library_root:
        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate_dir = library_root / f"{prefix}_{timestamp_tag}"
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            library_dir = candidate_dir.resolve()
        except OSError:
            library_dir = None

    def _export_to_library(label: str, source: Path | None) -> Path | None:
        if library_dir is None or source is None:
            return None
        try:
            source_path = Path(source).expanduser().resolve()
        except Exception:
            return None
        if not source_path.exists():
            return None
        target = library_dir / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.exists() or target.is_symlink():
                with contextlib.suppress(OSError):
                    if target.samefile(source_path):
                        shared_artifacts[label] = str(target)
                        return target
                    target.unlink()
        except OSError:
            with contextlib.suppress(OSError):
                if target.exists() or target.is_symlink():
                    target.unlink()
        try:
            shutil.copy2(str(source_path), str(target))
        except shutil.SameFileError:
            pass
        shared_artifacts[label] = str(target)
        return target

    report_dir = audio_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = raw_sources.get("report")
    if report_path and report_path.exists():
        report_target = report_dir / f"{prefix}_preprocessing_report.json"
        if report_path.resolve() != report_target.resolve():
            if report_target.exists():
                report_target.unlink()
            shutil.move(str(report_path), str(report_target))
        report_path = report_target
    else:
        report_path = None
    if report_path:
        _emit(
            f"[{total_artifacts}/{total_artifacts}] preprocessing report saved ({report_path.name}).",
            _artifact_progress(total_artifacts),
        )
    if library_dir:
        _emit("Copying audio artifacts to shared library…", 0.82)

    artifact_key_map: dict[str, tuple[str, ...]] = {
        "extracted": ("audio_extracted", "audio_raw"),
        "vocals": ("audio_vocals",),
        # Map new enhanced artifacts; also populate back-compat keys
        "enhanced_vocals": ("audio_enhanced_vocals", "audio_enhanced", "audio_processed"),
        "enhanced_mix": ("audio_enhanced_mix",),
        "report": ("preprocessing_report",),
    }

    artifact_updates: dict[str, str | None] = {}

    for source_key, metadata_keys in artifact_key_map.items():
        local_path = renamed.get(source_key) if source_key != "report" else report_path
        if local_path:
            rel_path = _relative_to_episode(episode_dir, Path(local_path))
            for metadata_key in metadata_keys:
                artifact_updates[metadata_key] = rel_path
        else:
            for metadata_key in metadata_keys:
                artifact_updates.setdefault(metadata_key, None)

    # Prefer canonical alias for processed pointer; otherwise the enhanced vocals path
    processed_canonical = canonical_aliases.get("audio_processed.wav")
    if processed_canonical:
        artifact_updates["audio_processed"] = _relative_to_episode(
            episode_dir, Path(processed_canonical)
        )
    else:
        enhanced_path = renamed.get("enhanced_vocals")
        if enhanced_path:
            artifact_updates["audio_processed"] = _relative_to_episode(
                episode_dir, Path(enhanced_path)
            )
        else:
            artifact_updates.setdefault("audio_processed", None)

    if library_dir:
        _export_to_library("audio_extracted", renamed.get("extracted"))
        _export_to_library("audio_vocals", renamed.get("vocals"))
        _export_to_library("audio_enhanced_vocals", renamed.get("enhanced_vocals"))
        _export_to_library("audio_enhanced_mix", renamed.get("enhanced_mix"))
        _export_to_library("audio_enhanced", renamed.get("enhanced_vocals"))
        _export_to_library("audio_processed", processed_canonical or renamed.get("enhanced_vocals"))
        _export_to_library("preprocessing_report", report_path)

    updates = {"artifacts": artifact_updates}
    updates["audio_library_dir"] = str(library_dir) if library_dir else None
    updates["shared_artifacts"] = shared_artifacts

    update_metadata(
        episode_dir,
        {
            **updates,
            "source_video_path": str(video_path),
            "preset": str(preset_path),
        },
    )
    _emit("Episode metadata updated with new audio artifacts.", 0.9)

    legacy_report = episode_dir / "preprocessing_report.json"
    if legacy_report.exists():
        if not report_path or legacy_report.resolve() != report_path.resolve():
            with contextlib.suppress(OSError):
                legacy_report.unlink()

    result = {
        "audio_extracted": canonical_aliases.get("audio_extracted.wav") or renamed.get("extracted"),
        "audio_vocals": canonical_aliases.get("audio_vocals.wav") or renamed.get("vocals"),
        "audio_enhanced_vocals": canonical_aliases.get("audio_enhanced_vocals.wav")
        or renamed.get("enhanced_vocals"),
        "audio_enhanced_mix": canonical_aliases.get("audio_enhanced_mix.wav")
        or renamed.get("enhanced_mix"),
        # Back-compat keys
        "audio_enhanced": canonical_aliases.get("audio_enhanced_vocals.wav")
        or renamed.get("enhanced_vocals"),
        "audio_processed": processed_canonical or renamed.get("enhanced_vocals"),
        "preprocessing_report": report_path,
        "audio_library_dir": library_dir,
    }

    if failure is not None:
        failure.artifacts = {
            "audio_extracted": result.get("audio_extracted"),
            "audio_vocals": result.get("audio_vocals"),
            "audio_enhanced_vocals": result.get("audio_enhanced_vocals"),
            "audio_enhanced_mix": result.get("audio_enhanced_mix"),
            "audio_enhanced": result.get("audio_enhanced"),
            "audio_processed": result.get("audio_processed"),
            "preprocessing_report": report_path,
            "audio_library_dir": library_dir,
        }
        raise failure

    _emit("Audio preprocessing stage complete.", 1.0)

    return result


def _run_stage_command(args: list[str], *, episode_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault(
        "PYTHONPATH", os.pathsep.join([str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")])
    )
    return subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run_transcription(
    episode_dir: Path,
    config_path: Path | None = None,
    *,
    audio_path_override: Path | None = None,
    progress: ProgressReporter | None = None,
) -> subprocess.CompletedProcess:
    """Trigger transcription for an episode."""

    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    progress_log_path = episode_dir / "transcription_progress.log"

    def _append_progress_log(message: str) -> None:
        try:
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{_timestamp()} {message}\n")
        except OSError:
            LOGGER.debug("Failed to write transcription progress log", exc_info=True)

    _emit("Validating audio artifacts…", 0.05)
    if audio_path_override is not None:
        audio_path = Path(audio_path_override).expanduser().resolve()
        if not audio_path.exists():
            _emit("Requested audio override not found; aborting.", 0.05)
            return subprocess.CompletedProcess(
                args=["transcription"],
                returncode=1,
                stdout="",
                stderr=f"Audio path not found: {audio_path}",
            )
    else:
        try:
            audio_path, _ = select_transcription_inputs(episode_dir)
        except FileNotFoundError as exc:
            _emit("Required audio artifacts missing; aborting transcription.", 0.05)
            return subprocess.CompletedProcess(
                args=["transcription"],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )

    environment = _current_environment()
    _emit("Initialising transcription pipeline…", 0.15)
    try:
        context = build_default_context(environment)
        transcriber = build_hybrid_transcriber(context.config, context.paths)
    except Exception as exc:
        _emit("Failed to initialise transcription pipeline.", 0.15)
        return subprocess.CompletedProcess(
            args=["transcription"],
            returncode=1,
            stdout="",
            stderr=f"Failed to initialise transcription pipeline: {exc}",
        )
    _emit("Transcription pipeline ready.", 0.3)

    show_config_candidate = config_path or metadata.get("show_config")
    _emit("Loading show configuration…", 0.35)
    show_config_path = _resolve_show_config_path(episode_dir, show_config_candidate)
    show_config = _load_json(show_config_path)

    options: TranscriptionOptions | None = None
    if isinstance(show_config, Mapping):
        prompt = show_config.get("transcription_prompt") or show_config.get("initial_prompt")
        if isinstance(prompt, str) and prompt.strip():
            options = TranscriptionOptions(initial_prompt=prompt.strip())

    _emit("Running transcription…", 0.45)

    init_hint: str | None = None
    local_transcriber = getattr(transcriber, "local", None)
    if isinstance(local_transcriber, LocalWhisperTranscriber):
        provider_settings = local_transcriber.provider_settings
        model_name = provider_settings.model
        compute_type = provider_settings.compute_type
        preferred_device = (provider_settings.device_preference or "auto").lower()
        device_hint = preferred_device
        runtime_preferences = getattr(local_transcriber, "_device_selector", None)
        priority_order: tuple[str, ...] | None = None
        if runtime_preferences is not None:
            priority_order = getattr(
                getattr(runtime_preferences, "runtime", None), "device_priority", None
            )
        if preferred_device == "auto" and priority_order:
            device_hint = priority_order[0]
        device_label = (device_hint or "auto").upper()
        init_hint = (
            "Decoder initialising – loading Whisper "
            f"{model_name} ({compute_type}) targeting {device_label}; "
            "first progress update appears after the model processes an initial segment."
        )
        _append_progress_log(
            "initialisation_hint "
            f"model={model_name} preferred_device={preferred_device} "
            f"device_priority={priority_order} compute_type={compute_type}"
        )
    else:
        init_hint = "Decoder initialising – preparing Whisper backend; first progress update appears after the initial segment."
        _append_progress_log("initialisation_hint backend=unknown")
    if init_hint:
        _emit(init_hint, None)

    def _transcribe_progress(elapsed_seconds: float, total_seconds: float | None) -> None:
        if total_seconds and total_seconds > 0:
            ratio = max(0.0, min(elapsed_seconds / total_seconds, 1.0))
            percent = ratio * 100.0
            stage_fraction = 0.45 + 0.25 * min(ratio, 0.99)
            message = (
                f"transcription_progress elapsed={elapsed_seconds:.2f}s "
                f"total={total_seconds:.2f}s percent={percent:.2f}"
            )
            _emit(f"Running transcription… ({percent:.1f}% complete)", stage_fraction)
        else:
            message = (
                f"transcription_progress elapsed={elapsed_seconds:.2f}s "
                f"total=unknown percent=unknown"
            )
            _emit(f"Running transcription… ({elapsed_seconds / 60:.1f} min elapsed)", None)
        LOGGER.info(message)
        _append_progress_log(message)
        safe_elapsed = max(0.0, float(elapsed_seconds))
        if total_seconds and total_seconds > 0:
            ratio = max(0.0, min(safe_elapsed / float(total_seconds), 1.0))
            stage_fraction = 0.45 + 0.25 * min(ratio, 0.99)
            _emit(
                f"Running transcription… ({ratio * 100:.1f}% complete)",
                stage_fraction,
            )
        else:
            _emit(
                f"Running transcription… ({safe_elapsed:.1f}s processed)",
                None,
            )

    try:
        result = transcriber.transcribe(
            str(audio_path),
            options=options,
            progress_callback=_transcribe_progress,
        )
    except Exception as exc:
        _emit("Transcription failed during inference.", 0.45)
        _append_progress_log(f"transcription_error {type(exc).__name__}: {exc}")
        LOGGER.exception("Transcription failed during inference.")
        return subprocess.CompletedProcess(
            args=["transcription"],
            returncode=1,
            stdout="",
            stderr=f"Transcription failed: {exc}",
        )
    _emit("Transcription completed; writing artifacts…", 0.75)

    transcript_payload = result.to_dict()
    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(episode_dir) or {}
    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        str(metadata.get("episode_id") or ""),
        metadata.get("created_at"),
    )
    transcript_path = transcripts_dir / f"{prefix}_transcript_raw.json"
    transcript_path.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")

    _emit("Updating episode metadata…", 0.85)
    metadata_updates: dict[str, Any] = {
        "artifacts": {"transcript_raw": _relative_to_episode(episode_dir, transcript_path)},
        "transcription": {"segments": len(transcript_payload.get("segments", []))},
    }
    if show_config_path is not None:
        metadata_updates["show_config"] = str(show_config_path)
    update_metadata(episode_dir, metadata_updates)
    _emit("Transcription metadata updated.", 0.9)

    segment_count = len(transcript_payload.get("segments", []))
    _emit("Transcription stage finished.", 1.0)
    stdout = f"Transcription completed with {segment_count} segments."
    return subprocess.CompletedProcess(
        args=["transcription"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def run_diarization(
    episode_dir: Path,
    config_path: Path | None = None,
    overrides: Mapping[str, float | int | None] | None = None,
) -> subprocess.CompletedProcess:
    """Trigger diarization for an episode."""

    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    try:
        audio_path, _ = select_transcription_inputs(episode_dir)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=["diarization"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )

    environment = _current_environment()
    try:
        context = build_default_context(environment)
        config_payload = copy.deepcopy(context.config)
        if not isinstance(config_payload, dict):  # pragma: no cover - defensive
            config_payload = dict(config_payload)
        if overrides:
            diar_cfg = config_payload.setdefault("diarization", {})
            if not isinstance(diar_cfg, dict):
                diar_cfg = {}
                config_payload["diarization"] = diar_cfg
            for key, value in overrides.items():
                if value is None:
                    diar_cfg.pop(key, None)
                else:
                    diar_cfg[key] = value
        diarizer = build_pyannote_diarizer(config_payload)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["diarization"],
            returncode=1,
            stdout="",
            stderr=f"Failed to initialise diarization pipeline: {exc}",
        )

    try:
        result = diarizer.diarize(str(audio_path))
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["diarization"],
            returncode=1,
            stdout="",
            stderr=f"Diarization failed: {exc}",
        )

    diarization_payload = result.to_dict()
    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(episode_dir) or {}
    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        str(metadata.get("episode_id") or ""),
        metadata.get("created_at"),
    )
    diarization_path = transcripts_dir / f"{prefix}_diarization.json"
    diarization_path.write_text(json.dumps(diarization_payload, indent=2), encoding="utf-8")

    metadata_updates = {
        "artifacts": {"diarization": _relative_to_episode(episode_dir, diarization_path)},
        "diarization": {
            "segments": len(diarization_payload.get("segments", [])),
            "speakers": diarization_payload.get("metadata", {}).get("speaker_count"),
        },
    }
    if metadata.get("show_config"):
        metadata_updates.setdefault("show_config", metadata.get("show_config"))
    update_metadata(episode_dir, metadata_updates)

    stdout = f"Diarization completed with {len(diarization_payload.get('segments', []))} segments."
    return subprocess.CompletedProcess(
        args=["diarization"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def run_dual_transcription(
    episode_dir: Path,
    config_path: Path | None = None,
    *,
    progress: ProgressReporter | None = None,
) -> subprocess.CompletedProcess:
    """Run ASR twice: once on enhanced vocals, once on enhanced mix.

    Produces:
        - transcript_raw.vocals.json (from audio_enhanced_vocals.wav)
        - transcript_raw.mix.json (from audio_enhanced_mix.wav)
    """
    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    # Resolve audio paths
    try:
        vocals_path, mix_path = select_transcription_inputs(episode_dir)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=["dual_transcription"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )

    environment = _current_environment()
    _emit("Initializing transcription pipeline…", 0.05)

    try:
        context = build_default_context(environment)
        transcriber = build_hybrid_transcriber(context.config, context.paths)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_transcription"],
            returncode=1,
            stdout="",
            stderr=f"Failed to initialize transcription pipeline: {exc}",
        )

    # Load show config for initial prompt
    show_config_candidate = config_path or metadata.get("show_config")
    show_config_path = _resolve_show_config_path(episode_dir, show_config_candidate)
    show_config = _load_json(show_config_path)

    options: TranscriptionOptions | None = None
    if isinstance(show_config, Mapping):
        prompt = show_config.get("transcription_prompt") or show_config.get("initial_prompt")
        if isinstance(prompt, str) and prompt.strip():
            options = TranscriptionOptions(initial_prompt=prompt.strip())

    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        str(metadata.get("episode_id") or ""),
        metadata.get("created_at"),
    )

    # Transcribe vocals
    _emit("Transcribing enhanced vocals…", 0.15)
    try:
        vocals_result = transcriber.transcribe(str(vocals_path), options=options)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_transcription"],
            returncode=1,
            stdout="",
            stderr=f"Vocals transcription failed: {exc}",
        )

    vocals_payload = vocals_result.to_dict()
    vocals_transcript_path = transcripts_dir / f"{prefix}_transcript_raw.vocals.json"
    vocals_transcript_path.write_text(json.dumps(vocals_payload, indent=2), encoding="utf-8")
    _emit("Vocals transcription complete", 0.5)

    # Transcribe mix
    _emit("Transcribing enhanced mix…", 0.55)
    try:
        mix_result = transcriber.transcribe(str(mix_path), options=options)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_transcription"],
            returncode=1,
            stdout="",
            stderr=f"Mix transcription failed: {exc}",
        )

    mix_payload = mix_result.to_dict()
    mix_transcript_path = transcripts_dir / f"{prefix}_transcript_raw.mix.json"
    mix_transcript_path.write_text(json.dumps(mix_payload, indent=2), encoding="utf-8")
    _emit("Mix transcription complete", 0.95)

    # Update metadata
    metadata_updates: dict[str, Any] = {
        "artifacts": {
            "transcript_raw_vocals": _relative_to_episode(episode_dir, vocals_transcript_path),
            "transcript_raw_mix": _relative_to_episode(episode_dir, mix_transcript_path),
        },
        "transcription": {
            "vocals_segments": len(vocals_payload.get("segments", [])),
            "mix_segments": len(mix_payload.get("segments", [])),
        },
    }
    if show_config_path is not None:
        metadata_updates["show_config"] = str(show_config_path)

    update_metadata(episode_dir, metadata_updates)
    _emit("Dual transcription complete", 1.0)

    stdout = (
        f"Dual transcription completed: "
        f"{len(vocals_payload.get('segments', []))} vocals segments, "
        f"{len(mix_payload.get('segments', []))} mix segments"
    )
    return subprocess.CompletedProcess(
        args=["dual_transcription"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def run_dual_alignment(
    episode_dir: Path,
    *,
    progress: ProgressReporter | None = None,
) -> subprocess.CompletedProcess:
    """Run diarization on vocals, then align both transcripts to it.

    Produces:
        - diarization.json (from audio_enhanced_vocals.wav)
        - aligned_vocals.jsonl (vocals transcript aligned to diarization)
        - aligned_mix.jsonl (mix transcript aligned to diarization)
    """
    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    # Check for transcripts
    vocals_transcript_path = resolve_artifact_path(episode_dir, "transcript_raw_vocals")
    mix_transcript_path = resolve_artifact_path(episode_dir, "transcript_raw_mix")

    if not vocals_transcript_path or not vocals_transcript_path.exists():
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr="Vocals transcript not found. Run dual transcription first.",
        )

    if not mix_transcript_path or not mix_transcript_path.exists():
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr="Mix transcript not found. Run dual transcription first.",
        )

    # Resolve vocals audio for diarization
    vocals_audio = resolve_artifact_path(episode_dir, "audio_enhanced_vocals")
    if not vocals_audio or not vocals_audio.exists():
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr="Enhanced vocals audio not found. Run CREATE AUDIO first.",
        )

    # Run diarization on vocals
    _emit("Running diarization on enhanced vocals…", 0.1)
    environment = _current_environment()
    try:
        context = build_default_context(environment)
        diarizer = build_pyannote_diarizer(context.config)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr=f"Failed to initialize diarization: {exc}",
        )

    try:
        diar_result = diarizer.diarize(str(vocals_audio))
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr=f"Diarization failed: {exc}",
        )

    _emit("Diarization complete", 0.3)

    # Save diarization
    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        str(metadata.get("episode_id") or ""),
        metadata.get("created_at"),
    )

    diarization_payload = diar_result.to_dict()
    diarization_path = transcripts_dir / f"{prefix}_diarization.json"
    diarization_path.write_text(json.dumps(diarization_payload, indent=2), encoding="utf-8")

    # Load transcripts
    vocals_payload = _load_json(vocals_transcript_path)
    mix_payload = _load_json(mix_transcript_path)

    if not isinstance(vocals_payload, Mapping) or not isinstance(mix_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr="Failed to parse transcript files",
        )

    vocals_transcription = _transcription_from_payload(vocals_payload)
    mix_transcription = _transcription_from_payload(mix_payload)
    diarization = _diarization_from_payload(diarization_payload)

    # Align both transcripts
    _emit("Aligning vocals transcript to diarization…", 0.4)

    try:
        from show_scribe.pipelines.alignment.align_asr_diar import (
            align_transcription_to_diarization,
        )
    except ImportError:
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr="Alignment module not found",
        )

    try:
        vocals_alignment = align_transcription_to_diarization(vocals_transcription, diarization)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr=f"Vocals alignment failed: {exc}",
        )

    _emit("Aligning mix transcript to diarization…", 0.7)

    try:
        mix_alignment = align_transcription_to_diarization(mix_transcription, diarization)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["dual_alignment"],
            returncode=1,
            stdout="",
            stderr=f"Mix alignment failed: {exc}",
        )

    # Save aligned outputs as JSONL
    vocals_aligned_path = transcripts_dir / f"{prefix}_aligned_vocals.jsonl"
    mix_aligned_path = transcripts_dir / f"{prefix}_aligned_mix.jsonl"

    with open(vocals_aligned_path, "w", encoding="utf-8") as f:
        for segment in vocals_alignment.segments:
            record = {
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker}
                    for w in segment.words
                ],
            }
            f.write(json.dumps(record) + "\n")

    with open(mix_aligned_path, "w", encoding="utf-8") as f:
        for segment in mix_alignment.segments:
            record = {
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker}
                    for w in segment.words
                ],
            }
            f.write(json.dumps(record) + "\n")

    _emit("Dual alignment complete", 0.95)

    # Update metadata
    update_metadata(
        episode_dir,
        {
            "artifacts": {
                "diarization": _relative_to_episode(episode_dir, diarization_path),
                "aligned_vocals": _relative_to_episode(episode_dir, vocals_aligned_path),
                "aligned_mix": _relative_to_episode(episode_dir, mix_aligned_path),
            },
            "diarization": {
                "segments": len(diarization_payload.get("segments", [])),
                "speakers": diarization_payload.get("metadata", {}).get("speaker_count"),
            },
        },
    )

    _emit("Complete", 1.0)

    stdout = (
        f"Dual alignment completed: {len(diarization_payload.get('segments', []))} diarization segments, "
        f"{len(vocals_alignment.segments)} vocals aligned segments, "
        f"{len(mix_alignment.segments)} mix aligned segments"
    )
    return subprocess.CompletedProcess(
        args=["dual_alignment"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def create_draft_transcript(
    episode_dir: Path,
    source: str = "vocals",
    *,
    progress: ProgressReporter | None = None,
) -> subprocess.CompletedProcess:
    """Create draft transcript files with numbered speakers (SPEAKER_00, SPEAKER_01, etc.).

    Args:
        episode_dir: Episode directory
        source: Which aligned source to use ("vocals" or "mix")
        progress: Optional progress callback

    Produces:
        - transcript_draft.txt
        - transcript_draft.srt
    """
    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    # Load aligned data
    aligned_key = f"aligned_{source}"
    aligned_path = resolve_artifact_path(episode_dir, aligned_key)

    if not aligned_path or not aligned_path.exists():
        return subprocess.CompletedProcess(
            args=["create_draft"],
            returncode=1,
            stdout="",
            stderr=f"Aligned {source} not found. Run dual alignment first.",
        )

    _emit(f"Loading aligned {source} data…", 0.1)

    # Read JSONL aligned data
    segments = []
    try:
        with open(aligned_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line))
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["create_draft"],
            returncode=1,
            stdout="",
            stderr=f"Failed to read aligned data: {exc}",
        )

    _emit("Building draft transcript with numbered speakers…", 0.3)

    # Collect unique speakers and create zero-padded mapping
    unique_speakers = []
    seen = set()
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            unique_speakers.append(speaker)
            seen.add(speaker)

    # Create speaker mapping with zero-padded numbers
    speaker_count = len(unique_speakers)
    padding = 2  # Always use 2 digits: SPEAKER_00, SPEAKER_01, etc.
    speaker_map = {}
    for idx, speaker in enumerate(unique_speakers):
        numbered_label = f"SPEAKER_{idx:0{padding}d}"
        speaker_map[speaker] = numbered_label

    # Build draft text
    draft_lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        numbered_speaker = speaker_map.get(speaker, "SPEAKER_XX")
        text = seg.get("text", "").strip()
        if text:
            draft_lines.append(f"{numbered_speaker}: {text}")

    draft_text = "\n\n".join(draft_lines)

    # Build draft SRT
    draft_srt_lines = []
    for idx, seg in enumerate(segments, start=1):
        speaker = seg.get("speaker", "UNKNOWN")
        numbered_speaker = speaker_map.get(speaker, "SPEAKER_XX")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if not text:
            continue

        # Format timestamps for SRT
        start_ts = _format_srt_timestamp(start)
        end_ts = _format_srt_timestamp(end)

        draft_srt_lines.append(f"{idx}")
        draft_srt_lines.append(f"{start_ts} --> {end_ts}")
        draft_srt_lines.append(f"[{numbered_speaker}] {text}")
        draft_srt_lines.append("")  # Empty line between entries

    draft_srt = "\n".join(draft_srt_lines)

    _emit("Writing draft files…", 0.8)

    # Save draft files
    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    draft_txt_path = transcripts_dir / "transcript_draft.txt"
    draft_srt_path = transcripts_dir / "transcript_draft.srt"

    draft_txt_path.write_text(draft_text, encoding="utf-8")
    draft_srt_path.write_text(draft_srt, encoding="utf-8")

    # Update metadata
    update_metadata(
        episode_dir,
        {
            "artifacts": {
                "transcript_draft_txt": _relative_to_episode(episode_dir, draft_txt_path),
                "transcript_draft_srt": _relative_to_episode(episode_dir, draft_srt_path),
            },
            "draft": {
                "source": source,
                "speaker_count": speaker_count,
                "segments": len(segments),
            },
        },
    )

    _emit("Draft transcript complete", 1.0)

    stdout = f"Draft transcript created from {source} with {speaker_count} numbered speakers and {len(segments)} segments"
    return subprocess.CompletedProcess(
        args=["create_draft"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_final_transcript(
    episode_dir: Path,
    speaker_mapping: dict[str, str],
    source: str = "vocals",
    *,
    progress: ProgressReporter | None = None,
) -> subprocess.CompletedProcess:
    """Create final transcript files with canonical speaker names.

    Args:
        episode_dir: Episode directory
        speaker_mapping: Map from numbered speakers (SPEAKER_00) to canonical names
        source: Which aligned source to use ("vocals" or "mix")
        progress: Optional progress callback

    Produces:
        - transcript_final.txt
        - transcript_final.srt
        - transcript_final.json
    """
    episode_dir = episode_dir.expanduser().resolve()

    def _emit(message: str, fraction: float | None = None) -> None:
        if progress is not None:
            progress(message, fraction)

    # Load aligned data
    aligned_key = f"aligned_{source}"
    aligned_path = resolve_artifact_path(episode_dir, aligned_key)

    if not aligned_path or not aligned_path.exists():
        return subprocess.CompletedProcess(
            args=["create_final"],
            returncode=1,
            stdout="",
            stderr=f"Aligned {source} not found. Run dual alignment first.",
        )

    _emit(f"Loading aligned {source} data…", 0.1)

    # Read JSONL aligned data
    segments = []
    try:
        with open(aligned_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line))
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["create_final"],
            returncode=1,
            stdout="",
            stderr=f"Failed to read aligned data: {exc}",
        )

    _emit("Applying speaker mapping to create final transcript…", 0.3)

    # Apply speaker mapping
    final_segments = []
    for seg in segments:
        numbered_speaker = seg.get("speaker", "UNKNOWN")
        canonical_speaker = speaker_mapping.get(numbered_speaker, numbered_speaker)
        final_segments.append(
            {
                **seg,
                "speaker": canonical_speaker,
            }
        )

    # Build final text
    final_lines = []
    for seg in final_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            final_lines.append(f"{speaker}: {text}")

    final_text = "\n\n".join(final_lines)

    # Build final SRT
    final_srt_lines = []
    for idx, seg in enumerate(final_segments, start=1):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if not text:
            continue

        # Format timestamps for SRT
        start_ts = _format_srt_timestamp(start)
        end_ts = _format_srt_timestamp(end)

        final_srt_lines.append(f"{idx}")
        final_srt_lines.append(f"{start_ts} --> {end_ts}")
        final_srt_lines.append(f"[{speaker}] {text}")
        final_srt_lines.append("")  # Empty line between entries

    final_srt = "\n".join(final_srt_lines)

    # Build final JSON
    final_json = {
        "metadata": {
            "episode_dir": str(episode_dir),
            "source": source,
            "speaker_mapping": speaker_mapping,
            "segment_count": len(final_segments),
            "unique_speakers": list(set(seg["speaker"] for seg in final_segments)),
        },
        "segments": final_segments,
    }

    _emit("Writing final transcript files…", 0.8)

    # Save final files
    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    final_txt_path = transcripts_dir / "transcript_final.txt"
    final_srt_path = transcripts_dir / "transcript_final.srt"
    final_json_path = transcripts_dir / "transcript_final.json"

    final_txt_path.write_text(final_text, encoding="utf-8")
    final_srt_path.write_text(final_srt, encoding="utf-8")
    final_json_path.write_text(json.dumps(final_json, indent=2), encoding="utf-8")

    # Update metadata
    update_metadata(
        episode_dir,
        {
            "artifacts": {
                "transcript_final_txt": _relative_to_episode(episode_dir, final_txt_path),
                "transcript_final_srt": _relative_to_episode(episode_dir, final_srt_path),
                "transcript_final": _relative_to_episode(episode_dir, final_json_path),
            },
            "final": {
                "source": source,
                "speaker_count": len(final_json["metadata"]["unique_speakers"]),
                "segments": len(final_segments),
                "speaker_mapping": speaker_mapping,
            },
        },
    )

    _emit("Final transcript complete", 1.0)

    unique_speakers = len(final_json["metadata"]["unique_speakers"])
    stdout = f"Final transcript created with {unique_speakers} canonical speakers and {len(final_segments)} segments"
    return subprocess.CompletedProcess(
        args=["create_final"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def run_align_and_export(episode_dir: Path) -> subprocess.CompletedProcess:
    """Align transcription with diarization and write transcript exports."""

    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    raw_path = resolve_artifact_path(episode_dir, "transcript_raw")
    if raw_path is None or not raw_path.exists():
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr="Raw transcript not found; run TRANSCRIPTION before alignment.",
        )

    diarization_path = resolve_artifact_path(episode_dir, "diarization")
    if diarization_path is None or not diarization_path.exists():
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr="Diarization artifact not found; run DIARIZATION before alignment.",
        )

    raw_payload = _load_json(raw_path)
    diar_payload = _load_json(diarization_path)
    if not isinstance(raw_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr="Unable to parse transcript_raw.json; rerun TRANSCRIPTION.",
        )
    if not isinstance(diar_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr="Unable to parse diarization results; rerun DIARIZATION.",
        )

    try:
        transcription = _transcription_from_payload(raw_payload)
        diarization = _diarization_from_payload(diar_payload)
    except Exception as exc:  # pragma: no cover - defensive parsing
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr=f"Failed to load transcription/diarization artifacts: {exc}",
        )

    environment = _current_environment()
    try:
        context = build_default_context(environment)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr=f"Failed to load configuration: {exc}",
        )

    alignment_options: dict[str, Any] = {}
    alignment_cfg = context.config.get("alignment") if isinstance(context.config, Mapping) else {}
    if isinstance(alignment_cfg, Mapping):
        alignment_options.update(dict(alignment_cfg))

    show_config_value = metadata.get("show_config")
    show_config_path = _resolve_show_config_path(episode_dir, show_config_value)
    if show_config_path is None:
        slug = metadata.get("show_slug")
        if isinstance(slug, str) and slug:
            candidate = context.paths.show_config_path(slug)
            if candidate.exists():
                show_config_path = candidate

    show_config: Mapping[str, Any] | None = _load_json(show_config_path)
    if isinstance(show_config, Mapping):
        overrides = show_config.get("alignment")
        if isinstance(overrides, Mapping):
            alignment_options.update(dict(overrides))

    auto_correct_names = True
    features_cfg = context.config.get("features") if isinstance(context.config, Mapping) else {}
    if isinstance(features_cfg, Mapping):
        auto_correct_names = bool(features_cfg.get("auto_correct_names", True))
    if isinstance(show_config, Mapping):
        auto_correct_names = bool(show_config.get("auto_correct_names", auto_correct_names))

    name_corrector = None
    if isinstance(show_config, Mapping):
        try:
            name_corrector = NameCorrector(show_config)
        except Exception:  # pragma: no cover - best effort
            name_corrector = None

    builder = TranscriptBuilder(name_corrector=name_corrector)
    pipeline = TranscriptPipeline(builder=builder, alignment_options=alignment_options)

    episode_id = str(metadata.get("episode_id") or episode_dir.name)
    show_display = str(metadata.get("show_display_name") or metadata.get("show_slug") or "")
    episode_metadata = {"episode_id": episode_id, "show": show_display}

    try:
        transcript_result = pipeline.run(
            transcription,
            diarization,
            episode_metadata=episode_metadata,
            auto_correct_names=auto_correct_names,
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["align_export"],
            returncode=1,
            stdout="",
            stderr=f"Alignment failed: {exc}",
        )

    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        episode_id,
        metadata.get("created_at"),
    )

    txt_canonical = transcripts_dir / "transcript_final.txt"
    srt_canonical = transcripts_dir / "transcript_final.srt"
    json_canonical = transcripts_dir / "transcript_final.json"

    txt_prefixed = transcripts_dir / f"{prefix}_transcript_final.txt"
    srt_prefixed = transcripts_dir / f"{prefix}_transcript_final.srt"
    json_prefixed = transcripts_dir / f"{prefix}_transcript_final.json"

    transcript_text = render_plain_text(transcript_result.document)
    txt_canonical.write_text(transcript_text, encoding="utf-8")
    if txt_prefixed != txt_canonical:
        txt_prefixed.write_text(transcript_text, encoding="utf-8")

    transcript_srt = render_srt(transcript_result.document)
    srt_canonical.write_text(transcript_srt, encoding="utf-8")
    if srt_prefixed != srt_canonical:
        srt_prefixed.write_text(transcript_srt, encoding="utf-8")

    payload = build_transcript_payload(
        transcript_result.document,
        alignment=transcript_result.alignment,
        corrections=transcript_result.corrections,
    )

    existing_created = None
    existing_updated = None
    if json_canonical.exists():
        try:
            existing_payload = json.loads(json_canonical.read_text(encoding="utf-8"))
            timestamps_block = existing_payload.get("metadata", {}).get("timestamps", {})
            existing_created = timestamps_block.get("created")
            existing_updated = timestamps_block.get("updated")
        except Exception:  # pragma: no cover - best effort
            existing_created = None
            existing_updated = None

    metadata_block = payload.setdefault("metadata", {})
    timestamps_block = metadata_block.setdefault("timestamps", {})
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    timestamps_block["updated"] = now
    if existing_created:
        timestamps_block["created"] = existing_created
    else:
        timestamps_block.setdefault("created", now)
    if existing_updated and existing_updated != timestamps_block["updated"]:
        timestamps_block.setdefault("previous_updated", existing_updated)

    json_payload = json.dumps(payload, indent=2)
    json_canonical.write_text(json_payload, encoding="utf-8")
    if json_prefixed != json_canonical:
        json_prefixed.write_text(json_payload, encoding="utf-8")

    update_metadata(
        episode_dir,
        {
            "artifacts": {
                "transcript_final": _relative_to_episode(episode_dir, json_canonical),
            },
            "transcript": {
                "segments": len(transcript_result.document.segments),
                "speakers": len(transcript_result.alignment.metadata.speakers),
            },
        },
    )

    stdout = (
        f"Alignment completed: {len(transcript_result.document.segments)} segments aligned "
        f"across {len(transcript_result.alignment.metadata.speakers)} speaker(s)."
    )
    return subprocess.CompletedProcess(
        args=["align_export"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def run_bleep_detection(episode_dir: Path) -> subprocess.CompletedProcess:
    """Trigger bleep detection for an episode."""

    episode_dir = episode_dir.expanduser().resolve()
    try:
        _, mix_path = select_transcription_inputs(episode_dir)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=["bleep_detection"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )

    return subprocess.CompletedProcess(
        args=["bleep_detection"],
        returncode=1,
        stdout="",
        stderr=(
            "Bleep detection pipeline is not yet implemented. "
            f"Expected enhanced mix at {mix_path}."
        ),
    )


def run_speaker_id(episode_dir: Path) -> subprocess.CompletedProcess:
    """Trigger speaker identification for an episode."""

    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}

    environment = _current_environment()
    try:
        context = build_default_context(environment)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=f"Failed to load configuration: {exc}",
        )

    features_cfg = context.config.get("features") if isinstance(context.config, Mapping) else {}
    if not isinstance(features_cfg, Mapping) or not bool(
        features_cfg.get("enable_voice_bank", False)
    ):
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Voice bank feature disabled in configuration; enable features.enable_voice_bank to proceed.",
        )

    try:
        audio_path, _ = select_transcription_inputs(episode_dir)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )

    diarization_path = resolve_artifact_path(episode_dir, "diarization")
    if diarization_path is None or not diarization_path.exists():
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Diarization artifact not found; run DIARIZATION before speaker identification.",
        )

    transcript_path = resolve_artifact_path(episode_dir, "transcript_final")
    alignment_payload = None
    transcript_payload = None

    if transcript_path and transcript_path.exists():
        # Legacy/final transcript exists
        transcript_payload = _load_json(transcript_path)
        if isinstance(transcript_payload, Mapping):
            alignment_payload = transcript_payload.get("alignment")
    else:
        # Fall back to dual transcription aligned JSONL files
        aligned_vocals_path = resolve_artifact_path(episode_dir, "aligned_vocals")
        aligned_mix_path = resolve_artifact_path(episode_dir, "aligned_mix")

        # Prefer vocals alignment for speaker ID
        aligned_path = (
            aligned_vocals_path
            if (aligned_vocals_path and aligned_vocals_path.exists())
            else aligned_mix_path
        )

        if not aligned_path or not aligned_path.exists():
            return subprocess.CompletedProcess(
                args=["speaker_identification"],
                returncode=1,
                stdout="",
                stderr="No transcript found; run ALIGN & EXPORT before speaker identification.",
            )

        # Load JSONL format (one segment per line)
        import json as _json

        segments = []
        try:
            with open(aligned_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        segments.append(_json.loads(line))
        except Exception as exc:
            return subprocess.CompletedProcess(
                args=["speaker_identification"],
                returncode=1,
                stdout="",
                stderr=f"Failed to parse aligned JSONL: {exc}",
            )

        # Construct alignment payload from JSONL segments
        alignment_payload = {"segments": segments}
        transcript_payload = {"alignment": alignment_payload, "segments": segments}
    diarization_payload = _load_json(diarization_path)
    if not isinstance(transcript_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Unable to parse transcript_final.json; regenerate the transcript exports.",
        )
    if not isinstance(diarization_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Unable to parse diarization results; rerun DIARIZATION.",
        )

    alignment_payload = transcript_payload.get("alignment")
    if not isinstance(alignment_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Transcript payload missing alignment data; rerun ALIGN & EXPORT.",
        )

    try:
        diarization = _diarization_from_payload(diarization_payload)
        alignment = _alignment_from_payload(alignment_payload)
    except Exception as exc:  # pragma: no cover - defensive parsing
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=f"Failed to load diarization/alignment artifacts: {exc}",
        )

    try:
        voice_pipeline = build_voice_bank_pipeline(context.config, context.paths)
    except SpeakerIdentificationError as exc:
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=f"Speaker identification unavailable: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=f"Failed to initialise voice bank pipeline: {exc}",
        )

    show_slug = str(metadata.get("show_slug") or "")
    episode_id = str(metadata.get("episode_id") or episode_dir.name)
    if not show_slug or not episode_id:
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr="Episode metadata missing show_slug or episode_id; re-run episode setup.",
        )

    try:
        result = voice_pipeline.identify(
            episode_id=episode_id,
            audio_path=audio_path,
            diarization=diarization,
            alignment=alignment,
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["speaker_identification"],
            returncode=1,
            stdout="",
            stderr=f"Speaker identification failed: {exc}",
        )

    result_dict = result.to_dict()
    assignments_by_cluster = {
        assignment["cluster_id"]: assignment
        for assignment in result_dict.get("assignments", [])
        if assignment.get("matched")
    }

    segments = list(transcript_payload.get("segments") or [])
    updated_transcript = False
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        metadata_block = segment.setdefault("metadata", {})
        original = metadata_block.get("original_speaker") or segment.get("speaker")
        if not isinstance(original, str) or not original:
            continue
        assignment = assignments_by_cluster.get(original)
        if not assignment:
            continue
        target_name = assignment.get("display_name") or assignment.get("speaker_key")
        if not target_name or segment.get("speaker") == target_name:
            continue
        metadata_block.setdefault("original_speaker", original)
        metadata_block["voice_bank_match"] = {
            key: assignment[key]
            for key in ("speaker_key", "display_name", "similarity", "auto_registered")
            if assignment.get(key) is not None
        }
        segment["speaker"] = target_name
        similarity = assignment.get("similarity")
        if similarity is not None:
            try:
                segment["speaker_confidence"] = float(similarity)
            except (TypeError, ValueError):
                pass
        updated_transcript = True

    if assignments_by_cluster:
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

    transcript_payload["segments"] = segments
    transcript_metadata = dict(transcript_payload.get("metadata") or {})
    transcript_metadata["voice_identification"] = result_dict
    if updated_transcript:
        transcript_metadata["speaker_order"] = _ordered_speakers(segments)
    transcript_payload["metadata"] = transcript_metadata
    transcript_payload["alignment"] = alignment.to_dict()

    transcript_path.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")

    update_metadata(
        episode_dir,
        {
            "artifacts": {
                "transcript_final": _relative_to_episode(episode_dir, transcript_path),
            },
            "voice_identification": {
                "matched": result.matched,
                "unmatched": result.unmatched,
            },
        },
    )

    stdout = (
        f"Speaker identification completed: matched {result.matched} cluster(s), "
        f"{result.unmatched} unmatched."
    )
    return subprocess.CompletedProcess(
        args=["speaker_identification"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def apply_manual_speaker_mapping(
    episode_dir: Path, mapping: Mapping[str, str]
) -> subprocess.CompletedProcess:
    """Apply a user-provided mapping from original speakers to target labels.

    Updates transcript_final.json by rewriting segment speakers and alignment speakers.
    Mapping keys should match original diarization cluster labels (e.g., "SPEAKER_00").
    """

    episode_dir = episode_dir.expanduser().resolve()
    transcript_path = resolve_artifact_path(episode_dir, "transcript_final")
    if transcript_path is None or not transcript_path.exists():
        return subprocess.CompletedProcess(
            args=["apply_manual_speaker_mapping"],
            returncode=1,
            stdout="",
            stderr="Final transcript not found; run ALIGN & EXPORT before manual mapping.",
        )

    transcript_payload = _load_json(transcript_path)
    if not isinstance(transcript_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["apply_manual_speaker_mapping"],
            returncode=1,
            stdout="",
            stderr="Unable to parse transcript_final.json; regenerate the transcript exports.",
        )

    if not mapping:
        return subprocess.CompletedProcess(
            args=["apply_manual_speaker_mapping"],
            returncode=1,
            stdout="",
            stderr="No speaker mapping provided.",
        )

    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            normalized[k] = v
    if not normalized:
        return subprocess.CompletedProcess(
            args=["apply_manual_speaker_mapping"],
            returncode=1,
            stdout="",
            stderr="No valid speaker mappings provided.",
        )

    segments = list(transcript_payload.get("segments") or [])
    updated_transcript = False
    updated_count = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        metadata_block = segment.setdefault("metadata", {})
        original = metadata_block.get("original_speaker") or segment.get("speaker")
        if not isinstance(original, str) or not original:
            continue
        target = normalized.get(original)
        if not target or segment.get("speaker") == target:
            continue
        metadata_block.setdefault("original_speaker", original)
        metadata_block["manual_label"] = True
        metadata_block["speaker_mapping"] = {"from": original, "to": target}
        segment["speaker"] = target
        updated_transcript = True
        updated_count += 1

    alignment_payload = transcript_payload.get("alignment")
    alignment: AlignmentResult | None = None
    if isinstance(alignment_payload, Mapping):
        try:
            alignment = _alignment_from_payload(alignment_payload)
        except Exception:
            alignment = None

    if alignment is not None:
        for aligned_segment in alignment.segments:
            original = aligned_segment.metadata.get("original_speaker") or aligned_segment.speaker
            target = normalized.get(original)
            if target and aligned_segment.speaker != target:
                aligned_segment.metadata.setdefault("original_speaker", original)
                aligned_segment.metadata["manual_label"] = True
                aligned_segment.speaker = target
                updated_transcript = True

        alignment.metadata.speakers = [
            normalized.get(name, name) for name in alignment.metadata.speakers
        ]
        transcript_payload["alignment"] = alignment.to_dict()

    if updated_transcript:
        transcript_payload["segments"] = segments
        transcript_metadata = dict(transcript_payload.get("metadata") or {})
        transcript_metadata["speaker_order"] = _ordered_speakers(segments)
        transcript_metadata.setdefault("voice_identification", {})
        transcript_metadata["voice_identification"]["manual_labels_applied"] = updated_count
        transcript_payload["metadata"] = transcript_metadata

        transcript_path.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")

        update_metadata(
            episode_dir,
            {
                "artifacts": {
                    "transcript_final": _relative_to_episode(episode_dir, transcript_path)
                },
                "voice_identification": {
                    "manual_labels_applied": updated_count,
                },
            },
        )

        return subprocess.CompletedProcess(
            args=["apply_manual_speaker_mapping"],
            returncode=0,
            stdout=f"Applied manual speaker mapping to {updated_count} segment(s).",
            stderr="",
        )

    return subprocess.CompletedProcess(
        args=["apply_manual_speaker_mapping"],
        returncode=0,
        stdout="No segments required updates; mapping may already be applied.",
        stderr="",
    )


def build_exports(episode_dir: Path) -> subprocess.CompletedProcess:
    """Build transcript exports for an episode."""

    episode_dir = episode_dir.expanduser().resolve()

    # Load transcript_final.json
    transcript_path = resolve_artifact_path(episode_dir, "transcript_final")
    if transcript_path is None or not transcript_path.exists():
        return subprocess.CompletedProcess(
            args=["build_exports"],
            returncode=1,
            stdout="",
            stderr="Final transcript not found; run alignment first.",
        )

    transcript_payload = _load_json(transcript_path)
    if not isinstance(transcript_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["build_exports"],
            returncode=1,
            stdout="",
            stderr="Unable to parse transcript_final.json.",
        )

    # Convert to TranscriptDocument
    segment_payloads = transcript_payload.get("segments") or []
    segments: list[TranscriptSegment] = []
    for seg_payload in segment_payloads:
        if not isinstance(seg_payload, Mapping):
            continue
        segments.append(
            TranscriptSegment(
                segment_id=int(seg_payload.get("id") or 0),
                start=float(seg_payload.get("start") or 0.0),
                end=float(seg_payload.get("end") or 0.0),
                speaker=str(seg_payload.get("speaker") or "Unknown"),
                text=str(seg_payload.get("text") or ""),
                speaker_confidence=seg_payload.get("speaker_confidence"),
                words=list(seg_payload.get("words") or []),
                metadata=dict(seg_payload.get("metadata") or {}),
            )
        )

    document = TranscriptDocument(
        segments=segments,
        metadata=dict(transcript_payload.get("metadata") or {}),
    )

    # Get metadata for file naming
    metadata = read_metadata(episode_dir) or {}
    prefix = metadata.get("artifact_prefix") or _build_artifact_prefix(
        str(metadata.get("show_slug") or ""),
        str(metadata.get("episode_id") or episode_dir.name),
        metadata.get("created_at"),
    )

    transcripts_dir = episode_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Generate exports
    txt_canonical = transcripts_dir / "transcript_final.txt"
    srt_canonical = transcripts_dir / "transcript_final.srt"
    txt_prefixed = transcripts_dir / f"{prefix}_transcript_final.txt"
    srt_prefixed = transcripts_dir / f"{prefix}_transcript_final.srt"

    transcript_text = render_plain_text(document)
    txt_canonical.write_text(transcript_text, encoding="utf-8")
    if txt_prefixed != txt_canonical:
        txt_prefixed.write_text(transcript_text, encoding="utf-8")

    transcript_srt = render_srt(document)
    srt_canonical.write_text(transcript_srt, encoding="utf-8")
    if srt_prefixed != srt_canonical:
        srt_prefixed.write_text(transcript_srt, encoding="utf-8")

    return subprocess.CompletedProcess(
        args=["build_exports"],
        returncode=0,
        stdout=f"Generated TXT and SRT exports with updated speaker names.",
        stderr="",
    )


def compute_analytics(episode_dir: Path) -> subprocess.CompletedProcess:
    """Compute analytics artifacts for an episode."""

    return subprocess.CompletedProcess(
        args=["compute_analytics"],
        returncode=1,
        stdout="",
        stderr="Analytics computation is not yet implemented.",
    )


def export_diarization_segments(
    episode_dir: Path,
    *,
    source_audio_key: str | None = None,
    limit: int | None = None,
) -> subprocess.CompletedProcess:
    """Export per-segment audio clips from diarization for manual speaker assignment.

    - Defaults to using `audio_enhanced_mix` (background preserved) if available,
      otherwise falls back to `audio_extracted`.
    - Writes files to `exports/speaker_review/<episode_id>/`.
    """

    episode_dir = episode_dir.expanduser().resolve()
    metadata = read_metadata(episode_dir) or {}
    episode_id = str(metadata.get("episode_id") or episode_dir.name)

    diar_path = resolve_artifact_path(episode_dir, "diarization")
    if diar_path is None or not diar_path.exists():
        return subprocess.CompletedProcess(
            args=["export_diar_segments"],
            returncode=1,
            stdout="",
            stderr="Diarization artifact not found; run DIARIZATION first.",
        )

    # Choose audio
    audio_path = None
    if source_audio_key:
        audio_path = resolve_artifact_path(episode_dir, source_audio_key)
    if audio_path is None:
        audio_path = resolve_artifact_path(
            episode_dir, "audio_enhanced_mix"
        ) or resolve_artifact_path(episode_dir, "audio_extracted")
    if audio_path is None or not audio_path.exists():
        return subprocess.CompletedProcess(
            args=["export_diar_segments"],
            returncode=1,
            stdout="",
            stderr="Unable to resolve source audio for snippet export.",
        )

    try:
        clip: AudioClip = _load_audio(audio_path, mono=True)
    except Exception as exc:  # pragma: no cover - runtime I/O
        return subprocess.CompletedProcess(
            args=["export_diar_segments"],
            returncode=1,
            stdout="",
            stderr=f"Failed to load audio: {exc}",
        )

    try:
        payload = json.loads(diar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return subprocess.CompletedProcess(
            args=["export_diar_segments"],
            returncode=1,
            stdout="",
            stderr=f"Failed to parse diarization JSON: {exc}",
        )

    segments = [seg for seg in payload.get("segments", []) if isinstance(seg, Mapping)]
    if not segments:
        return subprocess.CompletedProcess(
            args=["export_diar_segments"],
            returncode=1,
            stdout="",
            stderr="No diarization segments found.",
        )

    transcript_segments: list[dict[str, Any]] = []
    transcript_path = resolve_artifact_path(episode_dir, "transcript_final")
    if transcript_path is None:
        transcript_path = resolve_artifact_path(episode_dir, "transcript_raw")
    if transcript_path and transcript_path.exists():
        transcript_payload = _load_json(transcript_path) or {}
        for segment in transcript_payload.get("segments", []):
            if not isinstance(segment, Mapping):
                continue
            try:
                start_val = float(segment.get("start", 0.0))
                end_val = float(segment.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            transcript_segments.append(
                {
                    "start": start_val,
                    "end": end_val,
                    "text": str(segment.get("text", "")),
                    "speaker": segment.get("speaker"),
                }
            )

    def _match_transcript_entry(start_time: float, end_time: float) -> dict[str, Any] | None:
        best_entry: dict[str, Any] | None = None
        best_overlap = 0.0
        for transcript_entry in transcript_segments:
            entry_start = transcript_entry["start"]
            entry_end = transcript_entry["end"]
            if entry_end <= start_time or entry_start >= end_time:
                continue
            overlap = min(entry_end, end_time) - max(entry_start, start_time)
            if overlap > best_overlap:
                best_overlap = overlap
                best_entry = transcript_entry
        return best_entry

    export_root = PROJECT_ROOT / "exports" / "speaker_review"
    out_dir = export_root / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    clusters: dict[str, list[dict[str, Any]]] = {}
    count = 0
    for seg in segments:
        try:
            sid = int(seg.get("id", count))
        except (TypeError, ValueError):
            sid = count
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if limit is not None and count >= limit:
            break
        start_ms = int(round(start * 100))
        end_ms = int(round(end * 100))
        file_name = f"segment_{sid:04d}_{start_ms}-{end_ms}.wav"
        path = out_dir / file_name
        try:
            sub = _extract_segment(clip, start, end)
            _save_audio(sub, path, subtype="PCM_16", always_mono=True)
        except Exception:  # pragma: no cover - runtime I/O
            continue
        cluster_label = str(seg.get("speaker") or f"cluster_{sid}")
        matched_entry = _match_transcript_entry(start, end)
        matched_text = matched_entry.get("text") if isinstance(matched_entry, Mapping) else ""
        matched_speaker = (
            matched_entry.get("speaker") if isinstance(matched_entry, Mapping) else None
        )
        text_value = str(matched_text).strip() if isinstance(matched_text, str) else ""
        written.append(
            {
                "id": sid,
                "start": start,
                "end": end,
                "duration": end - start,
                "speaker": cluster_label,
                "assigned_speaker": matched_speaker,
                "path": str(path),
                "text": text_value,
            }
        )
        clusters.setdefault(cluster_label, []).append(written[-1])
        count += 1

    clusters_payload = {label: entries for label, entries in clusters.items()}

    summary = {
        "episode_id": episode_id,
        "source_audio": str(audio_path),
        "output_dir": str(out_dir),
        "segments": written,
        "clusters": clusters_payload,
        "created": _timestamp(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return subprocess.CompletedProcess(
        args=["export_diar_segments"],
        returncode=0,
        stdout=f"Exported {len(written)} segment(s) for review to {out_dir}.",
        stderr="",
    )


def assign_segment_speakers(
    episode_dir: Path, assignments: Sequence[Mapping[str, Any]]
) -> subprocess.CompletedProcess:
    """Apply manual speaker labels to individual transcript segments."""

    episode_dir = episode_dir.expanduser().resolve()

    if not assignments:
        return subprocess.CompletedProcess(
            args=["assign_segment_speakers"],
            returncode=1,
            stdout="",
            stderr="No segment assignments provided.",
        )

    # Load the ASR transcript (which has transcription text)
    # User assignments come from UI with diarization timestamps, but we'll match
    # them to ASR segments using overlap-based matching
    import json as _json

    transcript_path = resolve_artifact_path(episode_dir, "transcript_final")
    transcript_created = "final"
    transcript_payload = None

    if transcript_path and transcript_path.exists():
        # Final transcript exists - this has ASR segments with text
        transcript_payload = _load_json(transcript_path)
    else:
        # Try raw transcript (legacy single transcription)
        transcript_path = resolve_artifact_path(episode_dir, "transcript_raw")
        transcript_created = "raw"
        if transcript_path and transcript_path.exists():
            transcript_payload = _load_json(transcript_path)
        else:
            # Fall back to dual transcription aligned JSONL files
            aligned_vocals_path = resolve_artifact_path(episode_dir, "aligned_vocals")
            aligned_mix_path = resolve_artifact_path(episode_dir, "aligned_mix")

            # Prefer vocals alignment
            aligned_path = (
                aligned_vocals_path
                if (aligned_vocals_path and aligned_vocals_path.exists())
                else aligned_mix_path
            )

            if not aligned_path or not aligned_path.exists():
                return subprocess.CompletedProcess(
                    args=["assign_segment_speakers"],
                    returncode=1,
                    stdout="",
                    stderr="Transcript not found; run transcription and alignment first.",
                )

            # Load JSONL format
            segments = []
            try:
                with open(aligned_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            segments.append(_json.loads(line))
            except Exception as exc:
                return subprocess.CompletedProcess(
                    args=["assign_segment_speakers"],
                    returncode=1,
                    stdout="",
                    stderr=f"Failed to parse aligned JSONL: {exc}",
                )

            transcript_payload = {"segments": segments}
            # Create a new transcript_final.json file for manual assignments
            transcript_path = episode_dir / "transcripts" / "transcript_final.json"
            transcript_created = "final"
    if not isinstance(transcript_payload, Mapping):
        return subprocess.CompletedProcess(
            args=["assign_segment_speakers"],
            returncode=1,
            stdout="",
            stderr="Unable to parse transcript JSON; regenerate the transcript.",
        )

    segments = list(transcript_payload.get("segments") or [])
    if not segments:
        return subprocess.CompletedProcess(
            args=["assign_segment_speakers"],
            returncode=1,
            stdout="",
            stderr="Transcript contains no segments to update.",
        )

    def _best_matching_segment(start: float, end: float) -> dict[str, Any] | None:
        best_segment: dict[str, Any] | None = None
        best_overlap = 0.0
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            try:
                seg_start = float(segment.get("start", 0.0))
                seg_end = float(segment.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if seg_end <= start or seg_start >= end:
                continue
            overlap = min(seg_end, end) - max(seg_start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = segment
        return best_segment

    updated_count = 0
    normalized_assignments: list[tuple[float, float, str]] = []
    for entry in assignments:
        try:
            start = float(entry.get("start"))
            end = float(entry.get("end"))
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            continue
        speaker_value = str(entry.get("speaker", "")).strip()
        if not speaker_value or end <= start:
            continue
        normalized_assignments.append((start, end, speaker_value))

    if not normalized_assignments:
        return subprocess.CompletedProcess(
            args=["assign_segment_speakers"],
            returncode=1,
            stdout="",
            stderr="No valid assignments provided.",
        )

    applied_segments: list[tuple[float, float, str]] = []
    for start, end, speaker_value in normalized_assignments:
        target_segment = _best_matching_segment(start, end)
        if target_segment is None:
            continue
        metadata_block = (
            target_segment.setdefault("metadata", {}) if isinstance(target_segment, dict) else {}
        )
        original_speaker = metadata_block.get("original_speaker") or target_segment.get("speaker")
        if target_segment.get("speaker") == speaker_value:
            continue
        metadata_block.setdefault("original_speaker", original_speaker)
        metadata_block["manual_label"] = True
        metadata_block["speaker_mapping"] = {
            "from": original_speaker,
            "to": speaker_value,
            "segment_start": start,
            "segment_end": end,
        }
        target_segment["speaker"] = speaker_value
        if isinstance(target_segment, dict):
            target_segment["metadata"] = metadata_block
        updated_count += 1
        applied_segments.append((start, end, speaker_value))

    if updated_count == 0:
        return subprocess.CompletedProcess(
            args=["assign_segment_speakers"],
            returncode=0,
            stdout="No segments required updates; labels may already be applied.",
            stderr="",
        )

    transcript_payload["segments"] = segments
    transcript_metadata = dict(transcript_payload.get("metadata") or {})
    transcript_metadata.setdefault("voice_identification", {})
    manual_count = int(
        transcript_metadata["voice_identification"].get("manual_segments_assigned", 0)
    )
    transcript_metadata["voice_identification"]["manual_segments_assigned"] = (
        manual_count + updated_count
    )
    transcript_metadata["speaker_order"] = _ordered_speakers(segments)
    transcript_payload["metadata"] = transcript_metadata

    transcript_path.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")

    artifact_key = "transcript_final" if transcript_created == "final" else "transcript_raw"
    update_metadata(
        episode_dir,
        {
            "artifacts": {artifact_key: _relative_to_episode(episode_dir, transcript_path)},
            "voice_identification": {
                "manual_segments_assigned": manual_count + updated_count,
            },
        },
    )

    # Update summary.json, if present.
    metadata = read_metadata(episode_dir) or {}
    episode_id = str(metadata.get("episode_id") or episode_dir.name)
    summary_candidates = [
        PROJECT_ROOT / "exports" / "speaker_review" / episode_id / "summary.json",
        PROJECT_ROOT / "exports" / "speaker_review" / episode_dir.name / "summary.json",
    ]

    def _matches(entry_start: Any, entry_end: Any, start_val: float, end_val: float) -> bool:
        try:
            entry_start_f = float(entry_start)
            entry_end_f = float(entry_end)
        except (TypeError, ValueError):
            return False
        return abs(entry_start_f - start_val) < 0.05 and abs(entry_end_f - end_val) < 0.05

    for candidate in summary_candidates:
        if not candidate.exists():
            continue
        try:
            summary_payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary_changed = False

        segments_block = summary_payload.get("segments")
        if isinstance(segments_block, list):
            for segment_entry in segments_block:
                if not isinstance(segment_entry, dict):
                    continue
                for start, end, speaker_value in applied_segments:
                    if _matches(segment_entry.get("start"), segment_entry.get("end"), start, end):
                        if segment_entry.get("assigned_speaker") != speaker_value:
                            segment_entry["assigned_speaker"] = speaker_value
                            summary_changed = True

        clusters_block = summary_payload.get("clusters")
        if isinstance(clusters_block, Mapping):
            for cluster_entries in clusters_block.values():
                if not isinstance(cluster_entries, list):
                    continue
                for entry in cluster_entries:
                    if not isinstance(entry, dict):
                        continue
                    for start, end, speaker_value in applied_segments:
                        if _matches(entry.get("start"), entry.get("end"), start, end):
                            if entry.get("assigned_speaker") != speaker_value:
                                entry["assigned_speaker"] = speaker_value
                                summary_changed = True

        # Legacy format with top-level cluster keys
        for value in list(summary_payload.values()):
            if not isinstance(value, dict):
                continue
            snippets = value.get("snippets")
            if not isinstance(snippets, list):
                continue
            for entry in snippets:
                if not isinstance(entry, dict):
                    continue
                for start, end, speaker_value in applied_segments:
                    if _matches(entry.get("start"), entry.get("end"), start, end):
                        if entry.get("assigned_speaker") != speaker_value:
                            entry["assigned_speaker"] = speaker_value
                            summary_changed = True

        if summary_changed:
            summary_payload["updated"] = _timestamp()
            candidate.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    # Voice bank seeding: Generate and store embeddings for assigned segments
    ingestion_summary_lines: list[str] = []
    try:
        from ..config.load import load_config
        from ..storage.db import SQLiteDatabase
        from ..storage.paths import PathsConfig
        from ..storage.voice_bank_manager import VoiceBankManager, normalize_key
        from ..pipelines.voice_bank.ingest import ingest_segment_embedding, IngestionSummary
        from ..pipelines.embeddings.pyannote_embeddings import build_pyannote_embedding_backend

        # Load config
        config = load_config(env="balanced")  # Use active preset
        features_cfg = config.get("features", {})
        voice_bank_enabled = bool(features_cfg.get("enable_voice_bank", True))

        if voice_bank_enabled and applied_segments:
            # Initialize voice bank manager
            paths_cfg = PathsConfig(config)
            voice_bank_root = paths_cfg.data_root / "voice_bank"
            voice_bank_root.mkdir(parents=True, exist_ok=True)

            db = SQLiteDatabase(paths_cfg.voice_bank_db)
            if not db.db_path.exists():
                db.initialize()
            voice_bank_mgr = VoiceBankManager(db)

            # Build embedding backend
            try:
                embedding_backend = build_pyannote_embedding_backend(config)
            except Exception as exc:
                LOGGER.warning("Could not initialize embedding backend: %s", exc)
                embedding_backend = None

            if embedding_backend is not None:
                # Get enhanced vocals audio path
                enhanced_vocals_path = resolve_artifact_path(episode_dir, "audio_enhanced_vocals")
                if not enhanced_vocals_path or not enhanced_vocals_path.exists():
                    # Fall back to enhanced mix or raw
                    enhanced_vocals_path = resolve_artifact_path(episode_dir, "audio_enhanced_mix")
                if not enhanced_vocals_path or not enhanced_vocals_path.exists():
                    enhanced_vocals_path = resolve_artifact_path(episode_dir, "audio_extracted")

                if enhanced_vocals_path and enhanced_vocals_path.exists():
                    ingestion_summary = IngestionSummary()

                    # Group assignments by speaker
                    assignments_by_speaker: dict[str, list[tuple[float, float]]] = {}
                    for start, end, speaker_name in applied_segments:
                        assignments_by_speaker.setdefault(speaker_name, []).append((start, end))

                    # Process each speaker
                    for speaker_name, time_ranges in assignments_by_speaker.items():
                        # Get or create speaker profile
                        speaker_key = normalize_key(speaker_name)
                        profile = voice_bank_mgr.get_speaker_by_key(speaker_key)

                        if profile is None:
                            # Auto-create new profile
                            profile = voice_bank_mgr.upsert_speaker(
                                display_name=speaker_name,
                                key=speaker_key,
                                notes="Auto-created from manual speaker assignment",
                            )
                            LOGGER.info(
                                "Auto-created speaker profile: %s (key=%s)",
                                speaker_name,
                                speaker_key,
                            )

                        # Ingest each segment
                        for start, end in time_ranges:
                            result = ingest_segment_embedding(
                                audio_path=enhanced_vocals_path,
                                speaker_profile=profile,
                                start=start,
                                end=end,
                                voice_bank_manager=voice_bank_mgr,
                                embedding_backend=embedding_backend,
                                voice_bank_root=voice_bank_root,
                                config=config,
                                source_episode=episode_id,
                            )
                            ingestion_summary.results.append(result)

                    # Build summary report
                    summary_by_speaker = ingestion_summary.by_speaker()
                    if summary_by_speaker:
                        ingestion_summary_lines.append("\n\nVoice Bank Summary:")
                        for speaker_name, results in summary_by_speaker.items():
                            accepted = sum(1 for r in results if r.accepted)
                            rejected = sum(1 for r in results if not r.accepted)
                            profile = voice_bank_mgr.get_speaker_by_key(normalize_key(speaker_name))
                            total_embeddings = (
                                len(voice_bank_mgr.list_embeddings(profile.id))
                                if profile and profile.id
                                else 0
                            )

                            ingestion_summary_lines.append(
                                f"  {speaker_name}: +{accepted} embeddings, {rejected} rejected, {total_embeddings} total"
                            )

                            # Show rejection reasons
                            rejection_reasons = [
                                r.rejection_reason
                                for r in results
                                if not r.accepted and r.rejection_reason
                            ]
                            if rejection_reasons:
                                unique_reasons = list(dict.fromkeys(rejection_reasons))[
                                    :3
                                ]  # Show up to 3 unique reasons
                                for reason in unique_reasons:
                                    ingestion_summary_lines.append(f"    - {reason}")

    except Exception as exc:
        LOGGER.error("Voice bank seeding failed: %s", exc)
        ingestion_summary_lines.append(f"\n\nVoice bank seeding error: {exc}")

    stdout = f"Applied manual labels to {updated_count} segment(s) ({artifact_key})." + "".join(
        ingestion_summary_lines
    )
    return subprocess.CompletedProcess(
        args=["assign_segment_speakers"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def get_stage_status(episode_dir: Path) -> dict[str, bool]:
    """Return availability of key artifacts for checklist rendering."""

    episode_dir = episode_dir.expanduser()
    status: dict[str, bool] = {}
    for key in ARTIFACT_DEFAULTS:
        resolved = resolve_artifact_path(episode_dir, key)
        status[key] = bool(resolved and resolved.exists())
    return status
