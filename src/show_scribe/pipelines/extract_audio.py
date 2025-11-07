"""Stage 1 pipeline: extract, preprocess, and validate audio from source media files."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("soundfile is required for audio extraction pipeline operations.") from exc

from show_scribe.storage.naming import EpisodeDescriptor, resolve_artifact_path
from show_scribe.storage.paths import PathsConfig
from show_scribe.utils import audio_io
from show_scribe.utils.audio_io import AudioClip
from show_scribe.utils.ffmpeg import FFmpeg, FFmpegError, FFmpegProgress, LoudnessSettings

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}

ProgressCallback = Callable[[FFmpegProgress], None]

logger = logging.getLogger(__name__)


class AudioExtractionError(RuntimeError):
    """Raised when the extraction pipeline encounters an unrecoverable error."""


@dataclass(slots=True)
class QualityThresholds:
    """Threshold configuration for audio quality checks."""

    min_duration_seconds: float = 1.0
    min_peak_dbfs: float = -50.0
    min_rms: float = 1e-4
    enforce_strict: bool = False


@dataclass(slots=True)
class AudioExtractionConfig:
    """Parameters that control the audio extraction pipeline."""

    sample_rate: int
    channels: int
    codec: str
    output_extension: str
    loudness: LoudnessSettings
    thresholds: QualityThresholds
    enable_preprocessing: bool = False  # NEW: Enable preprocessing

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
        *,
        default_extension: str = "wav",
    ) -> AudioExtractionConfig:
        raw_audio = config.get("audio", {})
        if not isinstance(raw_audio, Mapping):
            raw_audio = {}
        audio_section = dict(raw_audio)

        raw_normalization = audio_section.get("normalization", {})
        if not isinstance(raw_normalization, Mapping):
            raw_normalization = {}
        normalization = dict(raw_normalization)

        raw_thresholds = config.get("audio_quality", {})
        if not isinstance(raw_thresholds, Mapping):
            raw_thresholds = {}
        thresholds_section = dict(raw_thresholds)

        # NEW: Check if preprocessing is enabled
        raw_preprocessing = config.get("audio_preprocessing", {})
        if not isinstance(raw_preprocessing, Mapping):
            raw_preprocessing = {}
        preprocessing_section = dict(raw_preprocessing)
        enable_preprocessing = bool(preprocessing_section.get("enable", False))

        loudness = LoudnessSettings(
            target_lufs=float(normalization.get("target_lufs", -20.0)),
            loudness_range=float(normalization.get("loudness_range", 7.0)),
            true_peak=float(normalization.get("true_peak", -1.0)),
            dual_pass=bool(normalization.get("dual_pass", False)),
        )

        thresholds = QualityThresholds(
            min_duration_seconds=float(thresholds_section.get("min_duration_seconds", 1.0)),
            min_peak_dbfs=float(thresholds_section.get("min_peak_dbfs", -50.0)),
            min_rms=float(thresholds_section.get("min_rms", 1e-4)),
            enforce_strict=bool(thresholds_section.get("enforce_strict", False)),
        )

        return cls(
            sample_rate=int(audio_section.get("sample_rate", 16_000)),
            channels=int(audio_section.get("channels", 1)),
            codec=str(audio_section.get("codec", "pcm_s16le")),
            output_extension=str(audio_section.get("format", default_extension)),
            loudness=loudness,
            thresholds=thresholds,
            enable_preprocessing=enable_preprocessing,
        )


@dataclass(slots=True)
class QualityCheckResult:
    """Outcome of a single audio quality check."""

    name: str
    passed: bool
    details: str


@dataclass(slots=True)
class AudioQualityReport:
    """Collection of quality checks performed on extracted audio."""

    checks: list[QualityCheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(slots=True)
class AudioExtractionResult:
    """Details about the extracted audio artifact."""

    audio_path: Path
    descriptor: EpisodeDescriptor
    clip: AudioClip
    quality_report: AudioQualityReport
    metadata: dict[str, object]
    preprocessing_report: dict[str, object] | None = None  # NEW
    preprocessing_report_path: Path | None = None  # NEW


class AudioExtractor:
    """Encapsulates the audio extraction workflow."""

    def __init__(
        self,
        paths: PathsConfig,
        config: AudioExtractionConfig,
        *,
        ffmpeg: FFmpeg | None = None,
        full_config: dict[str, object] | None = None,  # NEW: For preprocessor
    ) -> None:
        self.paths = paths
        self.config = config
        self.ffmpeg = ffmpeg or FFmpeg(loudness=config.loudness)
        self.full_config = full_config or {}

        # NEW: Initialize preprocessor if enabled
        self.preprocessor = None
        if config.enable_preprocessing:
            try:
                from show_scribe.pipelines.audio_preprocessing import AudioPreprocessor
            except ImportError:
                logger.warning(
                    "Audio preprocessing enabled but dependencies not installed. "
                    "Install with: pip install audio-separator resemble-enhance"
                )
            else:
                self.preprocessor = AudioPreprocessor(self.full_config)
                logger.info("Audio preprocessing enabled")

    def extract(
        self,
        source_media: str | Path,
        descriptor: EpisodeDescriptor,
        *,
        progress: ProgressCallback | None = None,
    ) -> AudioExtractionResult:
        """Extract and optionally preprocess audio for the given episode descriptor."""
        source_path = Path(source_media)
        self._validate_source(source_path)

        # Extract raw audio
        audio_path = resolve_artifact_path(
            self.paths,
            descriptor,
            "audio_extracted",
            self.config.output_extension,
            ensure_directory=True,
        )

        try:
            self.ffmpeg.extract_audio(
                source_path,
                audio_path,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                audio_codec=self.config.codec,
                progress=progress,
            )
        except FFmpegError as exc:
            raise AudioExtractionError(f"FFmpeg failed: {exc}") from exc

        # NEW: Apply preprocessing if enabled
        preprocessing_report = None
        preprocessing_report_path: Path | None = None
        if self.preprocessor is not None:
            logger.info("Applying audio preprocessing...")
            episode_dir = self.paths.episode_directory(descriptor.show_name, descriptor.episode_id)
            processed_dir = episode_dir / "processed_audio"
            processed_dir.mkdir(parents=True, exist_ok=True)

            try:
                artifacts = self.preprocessor.preprocess(
                    str(audio_path),
                    output_dir=str(processed_dir),
                )

                processed_path = resolve_artifact_path(
                    self.paths,
                    descriptor,
                    "audio_processed",
                    self.config.output_extension,
                    ensure_directory=True,
                )

                preferred_sources: tuple[Path | str | None, ...] = (
                    artifacts.enhanced_audio,
                    artifacts.enhanced_mix_audio,
                    artifacts.vocals_audio,
                    artifacts.final_audio,
                    audio_path,
                )
                selected_source: Path | None = None
                for candidate in preferred_sources:
                    if not candidate:
                        continue
                    try:
                        candidate_path = Path(candidate).expanduser().resolve()
                    except Exception:  # pragma: no cover - defensive
                        continue
                    if candidate_path.exists():
                        selected_source = candidate_path
                        break
                if selected_source is None:
                    selected_source = processed_path

                if selected_source != processed_path:
                    shutil.copy2(selected_source, processed_path)
                audio_path = processed_path

                preprocessing_report = dict(artifacts.report)
                preprocessing_report_path = artifacts.report_path
                files_section = dict(preprocessing_report.get("files", {}))
                files_section["processed"] = str(artifacts.final_audio)
                files_section["episode_artifact"] = str(processed_path)
                files_section["selected_for_transcription"] = str(selected_source)
                files_section["report"] = str(artifacts.report_path)
                files_section["vocals"] = (
                    str(artifacts.vocals_audio) if artifacts.vocals_audio else None
                )
                files_section["enhanced"] = (
                    str(artifacts.enhanced_audio) if artifacts.enhanced_audio else None
                )
                files_section["enhanced_vocals"] = (
                    str(artifacts.enhanced_audio) if artifacts.enhanced_audio else None
                )
                files_section["enhanced_mix"] = (
                    str(artifacts.enhanced_mix_audio) if artifacts.enhanced_mix_audio else None
                )
                preprocessing_report["files"] = files_section
                logger.info("Using preprocessed audio artifact: %s", audio_path)

            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Preprocessing failed: %s. Using original audio.", exc)
                preprocessing_report = {"error": str(exc), "fallback_to_original": True}

        clip = audio_io.load_audio(
            audio_path,
            target_sample_rate=self.config.sample_rate,
            mono=self.config.channels == 1,
        )
        metadata = self._read_audio_metadata(audio_path)

        self._validate_output(metadata, clip)
        report = self._run_quality_checks(clip)

        if not report.passed and self.config.thresholds.enforce_strict:
            details = ", ".join(check.details for check in report.checks if not check.passed)
            raise AudioExtractionError(f"Audio quality checks failed: {details}")

        return AudioExtractionResult(
            audio_path=audio_path,
            descriptor=descriptor,
            clip=clip,
            quality_report=report,
            metadata=metadata,
            preprocessing_report=preprocessing_report,
            preprocessing_report_path=preprocessing_report_path,
        )

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _validate_source(self, path: Path) -> None:
        if not path.exists():
            raise AudioExtractionError(f"Source media not found: {path}")
        if not path.is_file():
            raise AudioExtractionError(f"Source path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise AudioExtractionError(
                "Unsupported media format "
                f"'{path.suffix}'. Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

    def _read_audio_metadata(self, audio_path: Path) -> dict[str, object]:
        with sf.SoundFile(str(audio_path), "r") as handle:
            return {
                "samplerate": handle.samplerate,
                "channels": handle.channels,
                "format": handle.format,
                "subtype": handle.subtype,
                "frames": handle.frames,
                "duration_seconds": handle.frames / float(handle.samplerate or 1),
            }

    def _validate_output(self, metadata: dict[str, object], clip: AudioClip) -> None:
        expected_rate = self.config.sample_rate
        expected_channels = self.config.channels

        if metadata["samplerate"] != expected_rate:
            raise AudioExtractionError(
                f"Unexpected sample rate {metadata['samplerate']} (expected {expected_rate})."
            )
        if metadata["channels"] != expected_channels:
            raise AudioExtractionError(
                "Unexpected number of channels "
                f"{metadata['channels']} (expected {expected_channels})."
            )

        if clip.samples.size == 0:
            raise AudioExtractionError("Extracted audio is empty.")

    def _run_quality_checks(self, clip: AudioClip) -> AudioQualityReport:
        stats = audio_io.compute_stats(clip)
        checks: list[QualityCheckResult] = []

        duration_passed = stats["duration_seconds"] >= self.config.thresholds.min_duration_seconds
        checks.append(
            QualityCheckResult(
                name="duration",
                passed=duration_passed,
                details=(
                    f"duration={stats['duration_seconds']:.2f}s "
                    f"(min {self.config.thresholds.min_duration_seconds:.2f}s)"
                ),
            )
        )

        peak_dbfs = stats["peak_dbfs"]
        peak_passed = peak_dbfs is not None and peak_dbfs >= self.config.thresholds.min_peak_dbfs
        checks.append(
            QualityCheckResult(
                name="peak_dbfs",
                passed=bool(peak_passed),
                details=(
                    (
                        f"peak={peak_dbfs:.2f}dBFS "
                        f"(min {self.config.thresholds.min_peak_dbfs:.2f}dBFS)"
                    )
                    if peak_dbfs is not None
                    else "peak=undefined"
                ),
            )
        )

        rms = stats["rms"]
        rms_passed = rms >= self.config.thresholds.min_rms
        checks.append(
            QualityCheckResult(
                name="rms",
                passed=rms_passed,
                details=f"rms={rms:.6f} (min {self.config.thresholds.min_rms:.6f})",
            )
        )

        return AudioQualityReport(checks=checks)


def build_extractor(
    config: dict[str, object],
    paths: PathsConfig,
    *,
    ffmpeg: FFmpeg | None = None,
) -> AudioExtractor:
    """Factory that constructs an :class:`AudioExtractor` from raw config."""
    extraction_config = AudioExtractionConfig.from_config(config)
    return AudioExtractor(paths, extraction_config, ffmpeg=ffmpeg, full_config=config)
