"""Client for hosted Whisper-compatible APIs (OpenAI Whisper API)."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar

import requests
from requests import Response, Session

from ...utils.logging import get_logger
from .whisper_local import (
    SegmentTranscription,
    TranscriptionMetadata,
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSettings,
    WordTiming,
)

LOGGER = get_logger(__name__)

__all__ = [
    "WhisperAPIError",
    "WhisperAPISettings",
    "WhisperAPITranscriber",
    "build_whisper_api_transcriber",
]


def _maybe_float(value: object | None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: object | None, default: float = 0.0) -> float:
    maybe = _maybe_float(value)
    return maybe if maybe is not None else default


def _coerce_int(value: object | None, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return default
    return default


def _normalise_env_key(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    stripped = str(value).strip()
    return stripped or None


@dataclass(slots=True)
class WhisperAPISettings:
    """Static configuration for the Whisper API client."""

    api_base_url: str
    timeout_seconds: float
    max_retries: int
    model: str
    api_key_env: str
    cost_per_minute_usd: float
    organization_env: str | None = None
    project_env: str | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> WhisperAPISettings:
        provider = config.get("providers")
        if not isinstance(provider, Mapping):
            raise TypeError("Configuration missing 'providers' section.")

        api_cfg = provider.get("whisper_api")
        if not isinstance(api_cfg, Mapping):
            raise TypeError("Configuration missing 'providers.whisper_api' section.")

        return cls(
            api_base_url=str(api_cfg.get("api_base_url")),
            timeout_seconds=float(api_cfg.get("timeout_seconds", 600.0)),
            max_retries=int(api_cfg.get("max_retries", 2)),
            model=str(api_cfg.get("model", "whisper-1")),
            api_key_env=str(api_cfg.get("api_key_env", "OPENAI_API_KEY")),
            cost_per_minute_usd=float(api_cfg.get("cost_per_minute_usd", 0.006)),
            organization_env=_normalise_env_key(api_cfg.get("organization_env")),
            project_env=_normalise_env_key(api_cfg.get("project_env")),
        )


class WhisperAPIError(RuntimeError):
    """Raised when transcription via a hosted API fails."""


class WhisperAPITranscriber:
    """Wrapper that submits audio to the Whisper HTTP API and normalises responses."""

    RETRY_STATUS_CODES: ClassVar[set[int]] = {429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: WhisperAPISettings,
        transcription_settings: TranscriptionSettings,
        *,
        session: Session | None = None,
    ) -> None:
        self.settings = settings
        self.transcription_settings = transcription_settings
        self._session = session or requests.Session()

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        options: TranscriptionOptions | None = None,
        api_key: str | None = None,
    ) -> TranscriptionResult:
        """Submit audio to the configured API and return unified transcription output."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        opts = options or TranscriptionOptions()
        payload = self._build_payload(opts)
        headers = self._build_headers(api_key)

        with audio_path.open("rb") as audio_handle:
            files = {
                "file": (audio_path.name or "audio.wav", audio_handle, "application/octet-stream"),
            }
            response = self._post_with_retries(files=files, data=payload, headers=headers)

        data = self._parse_response(response)
        result = self._to_transcription_result(data, opts)
        result.metadata.cost_usd = self._estimate_cost(result)
        return result

    # ------------------------------------------------------------------
    # HTTP utilities
    # ------------------------------------------------------------------
    def _post_with_retries(
        self,
        *,
        files: Mapping[str, tuple[str, BinaryIO, str]] | None = None,
        data: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        attempts = 0
        backoff = 1.0
        last_error: Exception | None = None

        while attempts <= self.settings.max_retries:
            try:
                response = self._session.post(
                    self.settings.api_base_url,
                    timeout=self.settings.timeout_seconds,
                    files=files,
                    data=data,
                    headers=headers,
                )
            except requests.RequestException as exc:  # pragma: no cover - network dependent
                last_error = exc
                LOGGER.warning("Whisper API request failed (%s); retrying.", exc)
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in self.RETRY_STATUS_CODES:
                    raise WhisperAPIError(
                        f"Whisper API responded with status {response.status_code}: {response.text}"
                    )
                last_error = WhisperAPIError(
                    f"Received retryable status {response.status_code}: {response.text}"
                )
                LOGGER.warning(
                    "Whisper API returned %s; backing off for %.1fs.",
                    response.status_code,
                    backoff,
                )
            attempts += 1
            if attempts > self.settings.max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

        raise WhisperAPIError("Exceeded maximum retries for Whisper API") from last_error

    def _build_headers(self, explicit_api_key: str | None) -> dict[str, str]:
        api_key = explicit_api_key or os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise WhisperAPIError(
                "Whisper API key not available. Set the environment variable "
                f"{self.settings.api_key_env}."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        if self.settings.organization_env:
            organization_id = os.environ.get(self.settings.organization_env)
            if organization_id:
                headers["OpenAI-Organization"] = organization_id
        if self.settings.project_env:
            project_id = os.environ.get(self.settings.project_env)
            if project_id:
                headers["OpenAI-Project"] = project_id
        return headers

    def _build_payload(self, opts: TranscriptionOptions) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.settings.model,
            "response_format": "verbose_json",
        }

        language = opts.language or self.transcription_settings.default_language
        if language:
            payload["language"] = language

        temperature = (
            opts.temperature
            if opts.temperature is not None
            else self.transcription_settings.temperature
        )
        payload["temperature"] = float(temperature)

        prompt = opts.initial_prompt or self.transcription_settings.initial_prompt
        if prompt:
            payload["prompt"] = prompt

        want_words = (
            opts.word_timestamps
            if opts.word_timestamps is not None
            else self.transcription_settings.enable_word_timestamps
        )
        granularities: list[str] = ["segment"]
        if want_words:
            granularities.append("word")
        payload["timestamp_granularities"] = granularities
        return payload

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------
    def _parse_response(self, response: Response) -> dict[str, object]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise WhisperAPIError("Unable to decode Whisper API response as JSON.") from exc
        if not isinstance(data, Mapping):
            raise WhisperAPIError("Whisper API returned a non-object payload.")
        return {str(key): value for key, value in data.items()}

    def _to_transcription_result(
        self,
        payload: Mapping[str, object],
        opts: TranscriptionOptions,
    ) -> TranscriptionResult:
        raw_segments = payload.get("segments")
        segments: Iterable[object]
        if isinstance(raw_segments, Iterable):
            segments = raw_segments
        else:
            segments = ()

        threshold = (
            opts.word_confidence_threshold
            if opts.word_confidence_threshold is not None
            else self.transcription_settings.word_confidence_threshold
        )

        parsed_segments: list[SegmentTranscription] = []
        for raw in segments:
            if not isinstance(raw, Mapping):
                continue
            words = self._parse_words(raw.get("words"), threshold)
            confidence = self._segment_confidence(raw, words)
            text_raw = raw.get("text")
            parsed_segments.append(
                SegmentTranscription(
                    segment_id=_coerce_int(raw.get("id"), len(parsed_segments)),
                    start=_coerce_float(raw.get("start")),
                    end=_coerce_float(raw.get("end")),
                    text=str(text_raw) if text_raw is not None else "",
                    confidence=confidence,
                    words=words,
                )
            )

        language_raw = payload.get("language")
        language = str(language_raw) if isinstance(language_raw, str) and language_raw else None

        metadata = TranscriptionMetadata(
            language=language,
            duration=_maybe_float(payload.get("duration")),
            detected_language_probability=_maybe_float(payload.get("language_probability")),
        )

        return TranscriptionResult(segments=parsed_segments, metadata=metadata)

    def _estimate_cost(self, result: TranscriptionResult) -> float | None:
        duration = result.metadata.duration
        if duration is None and result.segments:
            duration = max(segment.end for segment in result.segments)
        if duration is None:
            return None
        minutes = duration / 60.0
        if minutes <= 0:
            return None
        cost = minutes * self.settings.cost_per_minute_usd
        return round(cost, 4)

    @staticmethod
    def _parse_words(raw_words: object, threshold: float) -> list[WordTiming]:
        if not isinstance(raw_words, Iterable):
            return []

        words: list[WordTiming] = []
        for entry in raw_words:
            if not isinstance(entry, Mapping):
                continue
            confidence_value = entry.get("confidence")
            probability_value = confidence_value
            if probability_value is None:
                probability_value = entry.get("probability")
            probability = _coerce_float(probability_value)
            if threshold and probability < threshold:
                continue
            word_raw = entry.get("word")
            words.append(
                WordTiming(
                    word=str(word_raw) if word_raw is not None else "",
                    start=_coerce_float(entry.get("start")),
                    end=_coerce_float(entry.get("end")),
                    probability=max(0.0, min(1.0, probability)),
                )
            )
        return words

    @staticmethod
    def _segment_confidence(raw_segment: Mapping[str, object], words: list[WordTiming]) -> float:
        if words:
            return sum(word.probability for word in words) / len(words)

        confidence_value = _maybe_float(raw_segment.get("confidence"))
        if confidence_value is not None:
            return max(0.0, min(1.0, confidence_value))

        avg_logprob_value = _maybe_float(raw_segment.get("avg_logprob"))
        if avg_logprob_value is not None:
            try:
                probability = math.exp(avg_logprob_value)
            except OverflowError:
                return 1.0
            return max(0.0, min(1.0, probability))
        return 0.0


def build_whisper_api_transcriber(
    config: Mapping[str, object],
    *,
    session: Session | None = None,
) -> WhisperAPITranscriber:
    """Factory helper that mirrors the local transcriber builder."""

    settings = WhisperAPISettings.from_config(config)
    transcription_settings = TranscriptionSettings.from_config(config)
    return WhisperAPITranscriber(settings, transcription_settings, session=session)
