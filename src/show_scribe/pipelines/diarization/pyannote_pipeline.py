"""Speaker diarization pipeline leveraging Pyannote."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ...utils.hf_hub_compat import ensure_use_auth_token_compat
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)
ensure_use_auth_token_compat()


def _ensure_hf_token_env(token: str) -> None:
    """Populate common Hugging Face auth environment variables when absent."""
    aliases = (
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "HF_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "PYANNOTE_TOKEN",
        "PYANNOTE_AUTH_TOKEN",
    )
    for env_var in aliases:
        if not os.environ.get(env_var):
            os.environ[env_var] = token


if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

try:  # pragma: no cover - optional dependency
    from pyannote.core import Annotation as RuntimeAnnotation
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    RuntimeAnnotation = None

_PyannotePipeline: type[Pipeline] | None
try:  # pragma: no cover - optional dependency
    from pyannote.audio import Pipeline as ImportedPipeline
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    _PyannotePipeline = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - optional dependency
    _PyannotePipeline = ImportedPipeline
    _IMPORT_ERROR = None

torch: ModuleType | None
try:  # pragma: no cover - optional dependency
    import torch as _torch_module
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
else:
    torch = _torch_module

__all__ = [
    "DevicePreferences",
    "DiarizationMetadata",
    "DiarizationResult",
    "DiarizationRuntimeOptions",
    "DiarizationSegment",
    "PyannoteDiarizer",
    "PyannoteNotAvailableError",
    "PyannoteProviderSettings",
    "build_pyannote_diarizer",
]


def _torch_feature_available(
    module: ModuleType | None,
    chain: Sequence[str],
    method: str,
) -> bool:
    current: object | None = module
    for attribute in chain:
        if current is None:
            return False
        current = getattr(current, attribute, None)
    if current is None:
        return False
    method_obj = getattr(current, method, None)
    if callable(method_obj):
        try:
            return bool(method_obj())
        except Exception:  # pragma: no cover - defensive
            return False
    return False


# ---------------------------------------------------------------------------
# Dataclasses capturing configuration and outputs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiarizationSegment:
    """A diarized segment with speaker attribution."""

    segment_id: int
    start: float
    end: float
    speaker: str
    track: str | int | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the segment to a dictionary."""
        payload: dict[str, Any] = {
            "id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
        }
        if self.track is not None:
            payload["track"] = self.track
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        payload["duration"] = max(self.end - self.start, 0.0)
        return payload


@dataclass(slots=True)
class DiarizationMetadata:
    """Metadata describing a diarization run."""

    model: str
    speaker_count: int
    duration: float | None
    inference_seconds: float | None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata to a dictionary."""
        return {
            "model": self.model,
            "speaker_count": self.speaker_count,
            "duration": self.duration,
            "inference_seconds": self.inference_seconds,
            "parameters": dict(self.parameters),
        }


@dataclass(slots=True)
class DiarizationResult:
    """Structured diarization output."""

    segments: list[DiarizationSegment]
    metadata: DiarizationMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialise the diarization result."""
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata.to_dict(),
        }

    @property
    def speakers(self) -> Sequence[str]:
        """Return the ordered list of unique speakers."""
        seen: set[str] = set()
        ordered: list[str] = []
        for segment in self.segments:
            if segment.speaker not in seen:
                seen.add(segment.speaker)
                ordered.append(segment.speaker)
        return ordered


@dataclass(slots=True)
class PyannoteProviderSettings:
    """Provider configuration drawn from the main project settings."""

    enabled: bool
    model: str
    auth_token_env: str
    segmentation_threshold: float | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PyannoteProviderSettings:
        providers = config.get("providers")
        if not isinstance(providers, Mapping):
            raise TypeError("Configuration missing 'providers' section.")

        pyannote_cfg = providers.get("pyannote")
        if not isinstance(pyannote_cfg, Mapping):
            raise TypeError("Configuration missing 'providers.pyannote' section.")

        enabled = bool(pyannote_cfg.get("enabled", True))
        model = str(pyannote_cfg.get("model", "pyannote/speaker-diarization@2.1"))
        auth_env = str(pyannote_cfg.get("auth_token_env", "SHOW_SCRIBE_PYANNOTE_TOKEN"))

        threshold_value = pyannote_cfg.get("segmentation_threshold")
        segmentation_threshold = float(threshold_value) if threshold_value is not None else None
        return cls(
            enabled=enabled,
            model=model,
            auth_token_env=auth_env,
            segmentation_threshold=segmentation_threshold,
        )


@dataclass(slots=True)
class DiarizationRuntimeOptions:
    """Runtime options that tailor diarization behaviour."""

    min_speakers: int | None
    max_speakers: int | None
    min_duration_on: float | None
    min_duration_off: float | None
    onset: float | None
    offset: float | None
    overlap_threshold: float | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DiarizationRuntimeOptions:
        diarization_cfg = config.get("diarization")
        if not isinstance(diarization_cfg, Mapping):
            diarization_cfg = {}

        def _int_or_none(key: str) -> int | None:
            value = diarization_cfg.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Configuration 'diarization.{key}' must be an integer if provided."
                ) from None

        def _float_or_none(key: str) -> float | None:
            value = diarization_cfg.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Configuration 'diarization.{key}' must be a float if provided."
                ) from None

        return cls(
            min_speakers=_int_or_none("min_speakers"),
            max_speakers=_int_or_none("max_speakers"),
            min_duration_on=_float_or_none("min_duration_on"),
            min_duration_off=_float_or_none("min_duration_off"),
            onset=_float_or_none("onset"),
            offset=_float_or_none("offset"),
            overlap_threshold=_float_or_none("overlap_threshold"),
        )


@dataclass(slots=True)
class DevicePreferences:
    """Device selection hints sourced from configuration."""

    prefer_gpu: bool
    device_priority: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DevicePreferences:
        runtime_cfg = config.get("runtime")
        if not isinstance(runtime_cfg, Mapping):
            runtime_cfg = {}
        prefer_gpu = bool(runtime_cfg.get("prefer_gpu", False))

        priority_raw = runtime_cfg.get("device_priority", ("cuda", "mps", "cpu"))
        if isinstance(priority_raw, Sequence):
            priority = tuple(str(entry).lower() for entry in priority_raw if str(entry).strip())
        else:
            priority = ("cuda", "mps", "cpu")
        return cls(prefer_gpu=prefer_gpu, device_priority=priority)

    def select_device(self) -> str:
        """Select the most appropriate device based on availability and preferences."""
        availability: dict[str, bool] = {
            "cuda": _torch_feature_available(torch, ("cuda",), "is_available"),
            "mps": _torch_feature_available(torch, ("backends", "mps"), "is_available"),
        }

        for candidate in self.device_priority:
            candidate = candidate.lower()
            if candidate == "cpu":
                return "cpu"
            if availability.get(candidate):
                return candidate

        if self.prefer_gpu:
            if availability.get("cuda"):
                return "cuda"
            if availability.get("mps"):
                return "mps"
        return "cpu"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PyannoteNotAvailableError(RuntimeError):
    """Raised when Pyannote is not available or configured incorrectly."""


# ---------------------------------------------------------------------------
# Pyannote diarizer implementation
# ---------------------------------------------------------------------------


class PyannoteDiarizer:
    """Facade around the pyannote.audio diarization pipeline."""

    def __init__(
        self,
        settings: PyannoteProviderSettings,
        runtime_options: DiarizationRuntimeOptions,
        *,
        device_preferences: DevicePreferences | None = None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("Pyannote diarization provider is disabled in configuration.")

        self.settings = settings
        self.runtime_options = runtime_options
        self.device_preferences = device_preferences or DevicePreferences(
            prefer_gpu=False, device_priority=("cpu",)
        )

        self._pipeline_lock = threading.Lock()
        self._pipeline: Pipeline | None = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diarize(
        self,
        audio_path: str | Path,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Run diarization on ``audio_path`` and return structured output."""

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        pipeline = self._ensure_pipeline_loaded()

        if min_speakers is None:
            min_speakers = self.runtime_options.min_speakers
        if max_speakers is None:
            max_speakers = self.runtime_options.max_speakers

        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers.")

        LOGGER.debug(
            "Running Pyannote diarization on %s (min_speakers=%s, max_speakers=%s).",
            path,
            min_speakers,
            max_speakers,
        )

        start_time = time.perf_counter()
        raw_output = pipeline(
            {"uri": path.stem, "audio": str(path)},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        annotation = self._unwrap_annotation(raw_output)
        inference_seconds = time.perf_counter() - start_time

        segments = self._annotation_to_segments(annotation)
        metadata = self._build_metadata(annotation, inference_seconds, min_speakers, max_speakers)
        return DiarizationResult(segments=segments, metadata=metadata)

    # ------------------------------------------------------------------
    # Pipeline loading and configuration
    # ------------------------------------------------------------------
    def _ensure_pipeline_loaded(self) -> Pipeline:
        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline

            if _PyannotePipeline is None:
                raise PyannoteNotAvailableError(
                    "pyannote.audio is not installed. Install the optional dependency "
                    "'pyannote.audio' to enable diarization."
                ) from _IMPORT_ERROR

            assert _PyannotePipeline is not None  # narrow for type checkers
            token = self._resolve_auth_token()
            if token:
                _ensure_hf_token_env(token)
            try:
                pipeline = _PyannotePipeline.from_pretrained(
                    self.settings.model,
                    use_auth_token=token,
                )
            except Exception as exc:
                raise PyannoteNotAvailableError(
                    f"Failed to load Pyannote pipeline '{self.settings.model}': {exc}"
                ) from exc
            if pipeline is None:
                raise PyannoteNotAvailableError(
                    "Failed to load Pyannote pipeline. Ensure the model identifier is correct "
                    "and that authentication is configured."
                )

            params: dict[str, Any] | None = None
            overrides_applied = False

            try:
                params = pipeline.parameters(instantiated=True)
            except Exception:  # pragma: no cover - defensive
                try:
                    params = pipeline.default_parameters()
                except Exception:
                    LOGGER.debug(
                        "Unable to load Pyannote instantiated parameters; will fall back to parameter space."
                    )
                    params = None

            if isinstance(params, dict):
                segmentation = params.get("segmentation")
                if isinstance(segmentation, dict):
                    overrides_applied |= self._apply_override(
                        segmentation,
                        "threshold",
                        self.settings.segmentation_threshold,
                    )
                    overrides_applied |= self._apply_override(
                        segmentation,
                        "min_duration_on",
                        self.runtime_options.min_duration_on,
                    )
                    overrides_applied |= self._apply_override(
                        segmentation,
                        "min_duration_off",
                        self.runtime_options.min_duration_off,
                    )
                    overrides_applied |= self._apply_override(
                        segmentation,
                        "onset",
                        self.runtime_options.onset,
                    )
                    overrides_applied |= self._apply_override(
                        segmentation,
                        "offset",
                        self.runtime_options.offset,
                    )

                clustering = params.get("clustering")
                if isinstance(clustering, dict):
                    overrides_applied |= self._apply_override(
                        clustering,
                        "overlap_rate",
                        self.runtime_options.overlap_threshold,
                    )
                    overrides_applied |= self._apply_override(
                        clustering,
                        "threshold",
                        self.runtime_options.overlap_threshold,
                    )

            try:
                if isinstance(params, dict):
                    pipeline.instantiate(params)
                    if overrides_applied:
                        LOGGER.debug("Applied custom Pyannote parameters: %s", params)
                else:
                    raise ValueError("Default parameters unavailable.")
            except ValueError as exc:
                LOGGER.warning(
                    "Default Pyannote parameters incompatible (%s); falling back to pipeline defaults without overrides.",
                    exc,
                )
                fallback_params = pipeline.parameters()
                pipeline.instantiate(fallback_params)

            self._device = self.device_preferences.select_device()
            move_to = getattr(pipeline, "to", None)
            if torch is not None and callable(move_to):
                device_ctor = getattr(torch, "device", None)
                if callable(device_ctor):
                    try:
                        move_to(device_ctor(self._device))
                    except Exception:  # pragma: no cover - device specific failures
                        LOGGER.debug(
                            "Unable to move Pyannote pipeline to device %s; falling back to CPU.",
                            self._device,
                        )
                        self._device = "cpu"

            self._pipeline = pipeline
            return self._pipeline

    @staticmethod
    def _apply_override(target: dict[str, Any], key: str, value: float | None) -> bool:
        if value is None or key not in target:
            return False
        try:
            target[key] = float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _resolve_auth_token(self) -> str:
        primary = self.settings.auth_token_env
        candidate_vars = [
            primary,
            "PYANNOTE_AUTH_TOKEN",
            "PYANNOTE_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGINGFACE_API_KEY",
            "HF_TOKEN",
            "HF_API_TOKEN",
        ]
        for env_var in candidate_vars:
            token = os.environ.get(env_var)
            if token:
                return token
        unique_vars = ", ".join(dict.fromkeys(candidate_vars))
        raise PyannoteNotAvailableError(
            "Pyannote authentication token not available. "
            f"Set one of the environment variables: {unique_vars}."
        )

    # ------------------------------------------------------------------
    # Annotation conversion helpers
    # ------------------------------------------------------------------
    def _unwrap_annotation(self, output: Any) -> Annotation:
        if RuntimeAnnotation is not None and isinstance(output, RuntimeAnnotation):
            return output
        if hasattr(output, "itertracks"):
            return output  # type: ignore[return-value]

        candidate = getattr(output, "speaker_diarization", None)
        if RuntimeAnnotation is not None and isinstance(candidate, RuntimeAnnotation):
            return candidate
        if isinstance(candidate, Mapping):
            inner = candidate.get("speaker_diarization")
            if RuntimeAnnotation is not None and isinstance(inner, RuntimeAnnotation):
                return inner
            if hasattr(inner, "itertracks"):
                return inner  # type: ignore[return-value]

        if isinstance(output, Mapping):
            mapped = output.get("speaker_diarization")
            if RuntimeAnnotation is not None and isinstance(mapped, RuntimeAnnotation):
                return mapped
            if hasattr(mapped, "itertracks"):
                return mapped  # type: ignore[return-value]

        raise PyannoteNotAvailableError(
            f"Unexpected diarization output type: {type(output)!r}. "
            "Ensure pyannote.audio >= 4.0 is installed."
        )

    def _annotation_to_segments(self, annotation: Annotation) -> list[DiarizationSegment]:
        entries: list[DiarizationSegment] = []
        try:
            iterator = annotation.itertracks(yield_label=True)
        except AttributeError:  # pragma: no cover - API defensive
            raise PyannoteNotAvailableError(
                "Unexpected annotation type returned by Pyannote pipeline."
            ) from None

        for idx, (segment, track, speaker_label) in enumerate(iterator):
            confidence = None
            try:
                details = annotation[segment, track]
            except Exception:  # pragma: no cover - defensive
                details = None
            if isinstance(details, Mapping):
                confidence_value = details.get("confidence")
                if confidence_value is not None:
                    try:
                        confidence = float(confidence_value)
                    except (TypeError, ValueError):
                        confidence = None

            entries.append(
                DiarizationSegment(
                    segment_id=idx,
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker_label),
                    track=track,
                    confidence=confidence,
                )
            )

        entries.sort(key=lambda item: (item.start, item.end))
        return entries

    def _build_metadata(
        self,
        annotation: Annotation,
        inference_seconds: float,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationMetadata:
        try:
            extent = annotation.get_timeline().extent()
            duration = float(extent.end - extent.start)
        except Exception:  # pragma: no cover - defensive
            duration = None

        parameters: dict[str, Any] = {
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "min_duration_on": self.runtime_options.min_duration_on,
            "min_duration_off": self.runtime_options.min_duration_off,
            "device": self._device,
            "onset": self.runtime_options.onset,
            "offset": self.runtime_options.offset,
            "overlap_threshold": self.runtime_options.overlap_threshold,
            "segmentation_threshold": self.settings.segmentation_threshold,
        }

        labels_method = getattr(annotation, "labels", None)
        if callable(labels_method):
            try:
                speaker_labels = list(labels_method())
            except Exception:  # pragma: no cover - defensive
                speaker_labels = []
        else:
            speaker_labels = []
        speaker_count = len(speaker_labels)

        return DiarizationMetadata(
            model=self.settings.model,
            speaker_count=speaker_count,
            duration=duration,
            inference_seconds=inference_seconds,
            parameters=parameters,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_pyannote_diarizer(config: Mapping[str, Any]) -> PyannoteDiarizer:
    """Return an instance of :class:`PyannoteDiarizer` configured from ``config``."""
    settings = PyannoteProviderSettings.from_config(config)
    runtime_options = DiarizationRuntimeOptions.from_config(config)
    device_preferences = DevicePreferences.from_config(config)
    return PyannoteDiarizer(
        settings=settings,
        runtime_options=runtime_options,
        device_preferences=device_preferences,
    )
