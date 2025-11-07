"""Local Faster-Whisper ASR integration."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...storage.paths import PathsConfig
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

_WhisperModel: Any | None

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel as _WhisperModel
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    _WhisperModel = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - optional dependency
    _IMPORT_ERROR = None

WhisperModel: Any = _WhisperModel

__all__ = [
    "LocalWhisperTranscriber",
    "RuntimePreferences",
    "SegmentTranscription",
    "TranscriptionMetadata",
    "TranscriptionOptions",
    "TranscriptionResult",
    "TranscriptionSettings",
    "WhisperModelLoader",
    "WhisperProviderSettings",
    "WordTiming",
    "build_whisper_transcriber",
]


@dataclass(slots=True)
class WordTiming:
    """Word-level timing information."""

    word: str
    start: float
    end: float
    probability: float


@dataclass(slots=True)
class SegmentTranscription:
    """A single transcribed segment."""

    segment_id: int
    start: float
    end: float
    text: str
    confidence: float
    words: list[WordTiming] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionMetadata:
    """Metadata about the transcription run."""

    language: str | None
    duration: float | None
    detected_language_probability: float | None = None
    cost_usd: float | None = None


@dataclass(slots=True)
class TranscriptionResult:
    """Container for segment data and metadata."""

    segments: list[SegmentTranscription]
    metadata: TranscriptionMetadata

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""
        return {
            "metadata": {
                "language": self.metadata.language,
                "duration": self.metadata.duration,
                "language_probability": self.metadata.detected_language_probability,
                "cost_usd": self.metadata.cost_usd,
            },
            "segments": [
                {
                    "id": segment.segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.confidence,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
        }


@dataclass(slots=True)
class TranscriptionOptions:
    """Per-call overrides for transcription."""

    language: str | None = None
    temperature: float | None = None
    initial_prompt: str | None = None
    beam_size: int | None = None
    word_timestamps: bool | None = None
    word_confidence_threshold: float | None = None
    vad_filter: bool | None = None
    device_override: str | None = None


@dataclass(slots=True)
class WhisperProviderSettings:
    """Provider-specific configuration for Faster-Whisper."""

    model: str
    compute_type: str
    device_preference: str
    download_root: Path
    beam_size: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        paths: PathsConfig,
    ) -> WhisperProviderSettings:
        providers = config.get("providers")
        if not isinstance(providers, Mapping):
            raise TypeError("Configuration missing 'providers' mapping.")
        provider = providers.get("whisper")
        if not isinstance(provider, Mapping):
            raise TypeError("Configuration 'providers.whisper' must be a mapping.")

        raw_root = provider.get("download_root", paths.models_dir / "whisper")
        download_root = Path(raw_root)
        if not download_root.is_absolute():
            download_root = paths.project_root / download_root
        download_root = download_root.expanduser().resolve()
        download_root.mkdir(parents=True, exist_ok=True)

        return cls(
            model=str(provider.get("model", "large-v3")),
            compute_type=str(provider.get("compute_type", "float16")),
            device_preference=str(provider.get("device", "auto")).lower(),
            download_root=download_root,
            beam_size=int(provider.get("beam_size", 5)),
        )


@dataclass(slots=True)
class RuntimePreferences:
    """Runtime device selection preferences."""

    prefer_gpu: bool
    device_priority: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> RuntimePreferences:
        runtime = config.get("runtime")
        if not isinstance(runtime, Mapping):
            raise TypeError("Configuration missing 'runtime' section.")
        prefer_gpu = bool(runtime.get("prefer_gpu", False))

        priority_raw = runtime.get("device_priority", ("cuda", "mps", "cpu"))
        if isinstance(priority_raw, Sequence):
            priority = tuple(str(entry).lower() for entry in priority_raw if str(entry).strip())
        else:
            priority = ("cuda", "mps", "cpu")
        return cls(prefer_gpu=prefer_gpu, device_priority=priority)


@dataclass(slots=True)
class TranscriptionSettings:
    """Default transcription settings drawn from configuration."""

    default_language: str | None
    temperature: float
    enable_word_timestamps: bool
    initial_prompt: str | None
    word_confidence_threshold: float

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> TranscriptionSettings:
        transcription = config.get("transcription")
        if not isinstance(transcription, Mapping):
            raise TypeError("Configuration missing 'transcription' section.")
        return cls(
            default_language=transcription.get("language"),
            temperature=float(transcription.get("temperature", 0.0)),
            enable_word_timestamps=bool(transcription.get("enable_word_timestamps", True)),
            initial_prompt=transcription.get("initial_prompt"),
            word_confidence_threshold=float(transcription.get("word_timestamps_threshold", 0.0)),
        )


class WhisperModelLoader:
    """Caches Faster-Whisper models in-memory."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str, Path], WhisperModel] = {}
        self._lock = threading.Lock()
        self._last_success: dict[tuple[str, str, str, Path], str] = {}

    def get(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: Path,
    ) -> WhisperModel:
        """Return a cached WhisperModel instance."""
        self._ensure_dependency_available()
        requested_key = (model_name, device, compute_type, download_root)

        with self._lock:
            last_error: Exception | None = None
            for candidate_compute in self._compute_type_candidates(compute_type):
                cache_key = (model_name, device, candidate_compute, download_root)
                if cache_key in self._cache:
                    self._last_success[requested_key] = candidate_compute
                    return self._cache[cache_key]

                try:
                    LOGGER.info(
                        "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s).",
                        model_name,
                        device,
                        candidate_compute,
                    )
                    model = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=candidate_compute,
                        download_root=str(download_root),
                    )
                except ValueError as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Failed to load Faster-Whisper model '%s' with compute_type=%s on device=%s: %s",
                        model_name,
                        candidate_compute,
                        device,
                        exc,
                    )
                    continue

                self._cache[cache_key] = model
                self._last_success[requested_key] = candidate_compute
                if candidate_compute != compute_type:
                    LOGGER.warning(
                        "Falling back to compute_type=%s for Faster-Whisper model '%s' on device=%s.",
                        candidate_compute,
                        model_name,
                        device,
                    )
                return model

        assert last_error is not None
        raise last_error

    def get_effective_compute_type(
        self,
        *,
        model_name: str,
        device: str,
        requested_compute_type: str,
        download_root: Path,
    ) -> str:
        key = (model_name, device, requested_compute_type, download_root)
        return self._last_success.get(key, requested_compute_type)

    @staticmethod
    def _compute_type_candidates(requested: str) -> list[str]:
        requested = requested or "float32"
        candidates: list[str] = [requested]
        if requested not in {"int8"}:
            candidates.append("int8")
        if "float32" not in candidates:
            candidates.append("float32")
        return candidates

    @staticmethod
    def _ensure_dependency_available() -> None:
        if WhisperModel is None:
            assert _IMPORT_ERROR is not None
            raise RuntimeError(
                "faster-whisper is not installed. Install it via "
                "'pip install faster-whisper' to enable local transcription."
            ) from _IMPORT_ERROR


class DeviceSelector:
    """Determines the best execution device based on configuration and availability."""

    def __init__(self, runtime: RuntimePreferences) -> None:
        self.runtime = runtime

    def select(self, preferred: str | None = None) -> str:
        """Return a device string understood by Faster-Whisper."""
        if preferred and preferred != "auto":
            resolved = self._resolve_specific(preferred)
            if resolved:
                return resolved

        for candidate in self._iter_candidates():
            resolved = self._resolve_specific(candidate)
            if resolved:
                return resolved
        return "cpu"

    def _iter_candidates(self) -> Iterable[str]:
        seen: set[str] = set()
        for device in self.runtime.device_priority:
            device = device.lower()
            if device in seen:
                continue
            seen.add(device)
            if device in {"cuda", "mps"} and not self.runtime.prefer_gpu:
                continue
            yield device
        yield "cpu"

    @staticmethod
    def _resolve_specific(device: str) -> str | None:
        device = device.lower()
        if device == "cuda" and _cuda_available():
            return "cuda"
        if device == "mps" and _mps_available():
            # Faster-Whisper does not expose a dedicated mps device; auto
            # allows CTranslate2 to pick the best available backend.
            return "auto"
        if device == "cpu":
            return "cpu"
        if device == "auto":
            return "auto"
        return None


class LocalWhisperTranscriber:
    """High-level wrapper around Faster-Whisper."""

    def __init__(
        self,
        provider_settings: WhisperProviderSettings,
        transcription_settings: TranscriptionSettings,
        runtime: RuntimePreferences,
        *,
        model_loader: WhisperModelLoader | None = None,
    ) -> None:
        self.provider_settings = provider_settings
        self.transcription_settings = transcription_settings
        self._device_selector = DeviceSelector(runtime)
        self._model_loader = model_loader or WhisperModelLoader()

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        options: TranscriptionOptions | None = None,
        progress_callback: Callable[[float, float | None], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio into text segments."""
        opts = options or TranscriptionOptions()
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        device = self._choose_device(opts)
        model = self._model_loader.get(
            model_name=self.provider_settings.model,
            device=device,
            compute_type=self.provider_settings.compute_type,
            download_root=self.provider_settings.download_root,
        )
        effective_compute = self._model_loader.get_effective_compute_type(
            model_name=self.provider_settings.model,
            device=device,
            requested_compute_type=self.provider_settings.compute_type,
            download_root=self.provider_settings.download_root,
        )
        if effective_compute != self.provider_settings.compute_type:
            LOGGER.info(
                "Adjusted Faster-Whisper compute_type from %s to %s for device=%s.",
                self.provider_settings.compute_type,
                effective_compute,
                device,
            )
            self.provider_settings.compute_type = effective_compute

        decode_options = self._build_decode_options(opts)
        segments_iter, info = model.transcribe(str(audio_path), **decode_options)
        total_duration = 0.0
        if info is not None:
            duration_attr = getattr(info, "duration", None)
            if duration_attr is not None:
                try:
                    total_duration = float(duration_attr)
                except (TypeError, ValueError):
                    total_duration = 0.0
        if total_duration <= 0:
            try:  # pragma: no cover - optional dependency
                import soundfile as sf

                with sf.SoundFile(str(audio_path)) as handle:
                    if handle.samplerate > 0:
                        total_duration = max(total_duration, handle.frames / handle.samplerate)
            except Exception:
                total_duration = max(0.0, total_duration)
        segments = self._collect_segments(
            segments_iter,
            opts,
            total_duration=total_duration if total_duration > 0 else None,
            progress_callback=progress_callback,
        )
        metadata = self._build_metadata(info)
        return TranscriptionResult(segments=segments, metadata=metadata)

    def _choose_device(self, opts: TranscriptionOptions) -> str:
        return self._device_selector.select(
            opts.device_override or self.provider_settings.device_preference
        )

    def _build_decode_options(self, opts: TranscriptionOptions) -> MutableMapping[str, object]:
        decode: MutableMapping[str, object] = {
            "beam_size": opts.beam_size or self.provider_settings.beam_size,
            "temperature": (
                opts.temperature
                if opts.temperature is not None
                else self.transcription_settings.temperature
            ),
            "word_timestamps": (
                opts.word_timestamps
                if opts.word_timestamps is not None
                else self.transcription_settings.enable_word_timestamps
            ),
            "initial_prompt": opts.initial_prompt or self.transcription_settings.initial_prompt,
        }
        language = opts.language or self.transcription_settings.default_language
        if language:
            decode["language"] = language
        if opts.vad_filter is not None:
            decode["vad_filter"] = opts.vad_filter
        return decode

    def _collect_segments(
        self,
        segments_iter: Iterator[object],
        opts: TranscriptionOptions,
        *,
        total_duration: float | None = None,
        progress_callback: Callable[[float, float | None], None] | None = None,
    ) -> list[SegmentTranscription]:
        threshold = (
            opts.word_confidence_threshold
            if opts.word_confidence_threshold is not None
            else self.transcription_settings.word_confidence_threshold
        )

        collected: list[SegmentTranscription] = []
        for raw_segment in segments_iter:
            segment_id = getattr(raw_segment, "id", len(collected))
            start = float(getattr(raw_segment, "start", 0.0))
            end = float(getattr(raw_segment, "end", 0.0))
            text = str(getattr(raw_segment, "text", "")).strip()
            words = self._extract_words(raw_segment, threshold)
            confidence = self._compute_confidence(raw_segment, words)
            collected.append(
                SegmentTranscription(
                    segment_id=int(segment_id),
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                    words=words,
                )
            )
            if progress_callback is not None:
                try:
                    progress_callback(end, total_duration)
                except Exception:  # pragma: no cover - progress callback errors should not crash
                    LOGGER.debug("Progress callback raised an exception", exc_info=True)
        return collected

    @staticmethod
    def _extract_words(raw_segment: object, threshold: float) -> list[WordTiming]:
        raw_words = getattr(raw_segment, "words", None)
        if not raw_words:
            return []

        words: list[WordTiming] = []
        for raw_word in raw_words:
            probability = float(getattr(raw_word, "probability", 0.0) or 0.0)
            if threshold and probability < threshold:
                continue
            words.append(
                WordTiming(
                    word=str(getattr(raw_word, "word", "")).strip(),
                    start=float(getattr(raw_word, "start", 0.0)),
                    end=float(getattr(raw_word, "end", 0.0)),
                    probability=probability,
                )
            )
        return words

    @staticmethod
    def _compute_confidence(raw_segment: object, words: list[WordTiming]) -> float:
        if words:
            average = sum(word.probability for word in words) / len(words)
            return _clamp_probability(average)

        avg_logprob = getattr(raw_segment, "avg_logprob", None)
        if avg_logprob is None:
            return 0.0
        try:
            return _clamp_probability(math.exp(float(avg_logprob)))
        except (TypeError, ValueError, OverflowError):
            return 0.0

    @staticmethod
    def _build_metadata(info: object | None) -> TranscriptionMetadata:
        if info is None:
            return TranscriptionMetadata(language=None, duration=None)

        language = getattr(info, "language", None)
        duration = getattr(info, "duration", None)
        probability = getattr(info, "language_probability", None)

        return TranscriptionMetadata(
            language=str(language) if language else None,
            duration=float(duration) if duration is not None else None,
            detected_language_probability=float(probability) if probability is not None else None,
        )


def build_whisper_transcriber(
    config: Mapping[str, object],
    paths: PathsConfig,
    *,
    model_loader: WhisperModelLoader | None = None,
) -> LocalWhisperTranscriber:
    """Factory helper to construct a LocalWhisperTranscriber."""
    provider_settings = WhisperProviderSettings.from_config(config, paths)
    transcription_settings = TranscriptionSettings.from_config(config)
    runtime = RuntimePreferences.from_config(config)
    return LocalWhisperTranscriber(
        provider_settings,
        transcription_settings,
        runtime,
        model_loader=model_loader,
    )


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cuda_available() -> bool:
    try:  # pragma: no cover - depends on environment
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch not installed
        return False


def _mps_available() -> bool:
    try:  # pragma: no cover - depends on environment
        import torch

        backend = getattr(torch, "backends", None)
        if backend and hasattr(backend, "mps"):
            return bool(torch.backends.mps.is_available())
    except Exception:
        return False
    return False
