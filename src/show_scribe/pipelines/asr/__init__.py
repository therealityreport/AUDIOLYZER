"""ASR provider factories and orchestration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from requests import Session

from ...storage.paths import PathsConfig
from ...utils.logging import get_logger
from .cost_tracking import record_asr_cost
from .whisper_api import (
    WhisperAPIError,
    WhisperAPITranscriber,
    build_whisper_api_transcriber,
)
from .whisper_local import (
    LocalWhisperTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
    WhisperModelLoader,
    build_whisper_transcriber,
)

LOGGER = get_logger(__name__)

__all__ = [
    "HybridWhisperTranscriber",
    "LocalWhisperTranscriber",
    "TranscriptionOptions",
    "TranscriptionResult",
    "WhisperAPIError",
    "WhisperAPITranscriber",
    "build_hybrid_transcriber",
    "build_whisper_api_transcriber",
    "build_whisper_transcriber",
]


class HybridWhisperTranscriber:
    """Transcriber that prefers the hosted API but falls back to local inference."""

    def __init__(
        self,
        config: Mapping[str, object],
        paths: PathsConfig | None,
        *,
        session: Session | None = None,
        model_loader: WhisperModelLoader | None = None,
        local: LocalWhisperTranscriber | None = None,
        remote: WhisperAPITranscriber | None = None,
    ) -> None:
        providers = config.get("providers", {})
        whisper_cfg = providers.get("whisper", {}) if isinstance(providers, Mapping) else {}
        self.mode = str(whisper_cfg.get("mode", "local")).lower()

        self._remote: WhisperAPITranscriber | None = None
        if local is not None:
            self._local = local
        else:
            if paths is None:
                raise ValueError("PathsConfig is required when building the local transcriber.")
            self._local = build_whisper_transcriber(
                config,
                paths,
                model_loader=model_loader,
            )

        if self.mode == "api":
            self._remote = remote or build_whisper_api_transcriber(config, session=session)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        options: TranscriptionOptions | None = None,
        api_key: str | None = None,
        progress_callback: Callable[[float, float | None], None] | None = None,
    ) -> TranscriptionResult:
        media_path = str(audio_path)

        if self.mode != "api" or self._remote is None:
            return self._local.transcribe(
                audio_path, options=options, progress_callback=progress_callback
            )

        try:
            result = self._remote.transcribe(audio_path, options=options, api_key=api_key)
        except WhisperAPIError as exc:
            LOGGER.warning(
                "Whisper API failed (%s); falling back to local Faster-Whisper.",
                exc,
            )
            return self._local.transcribe(
                audio_path, options=options, progress_callback=progress_callback
            )
        else:
            remote_settings = getattr(self._remote, "settings", None)
            if remote_settings is not None:
                record_asr_cost(
                    provider="whisper_api",
                    model=getattr(remote_settings, "model", ""),
                    cost_usd=result.metadata.cost_usd,
                    duration_seconds=result.metadata.duration,
                    media_path=media_path,
                    metadata={
                        "api_base_url": getattr(remote_settings, "api_base_url", ""),
                        "timeout_seconds": getattr(remote_settings, "timeout_seconds", None),
                    },
                )
            return result

    @property
    def local(self) -> LocalWhisperTranscriber:
        return self._local

    @property
    def remote(self) -> WhisperAPITranscriber | None:
        return self._remote


def build_hybrid_transcriber(
    config: Mapping[str, object],
    paths: PathsConfig,
    *,
    session: Session | None = None,
    model_loader: WhisperModelLoader | None = None,
) -> HybridWhisperTranscriber:
    """Construct a hybrid transcriber that can fall back to local inference."""

    return HybridWhisperTranscriber(
        config,
        paths,
        session=session,
        model_loader=model_loader,
    )
