"""Orchestrates transcript assembly from ASR, diarization, and alignment outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from ...utils.logging import get_logger
from ...utils.name_correction import NameCorrectionResult
from ..alignment.align_asr_diar import AlignmentResult, align_transcription_to_diarization
from ..asr.whisper_local import TranscriptionResult
from ..diarization.pyannote_pipeline import DiarizationResult
from .builder import TranscriptBuilder, TranscriptDocument

LOGGER = get_logger(__name__)

__all__ = [
    "TranscriptPipeline",
    "TranscriptPipelineResult",
]


@dataclass(slots=True)
class TranscriptPipelineResult:
    """Artifacts produced when assembling a transcript."""

    alignment: AlignmentResult
    document: TranscriptDocument
    corrections: list[NameCorrectionResult]


class TranscriptPipeline:
    """High-level coordinator for transcript generation."""

    def __init__(
        self,
        *,
        builder: TranscriptBuilder | None = None,
        alignment_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.builder = builder or TranscriptBuilder()
        self._alignment_options = dict(alignment_options or {})

    def run(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
        *,
        episode_metadata: Mapping[str, Any] | None = None,
        auto_correct_names: bool = True,
        alignment_overrides: Mapping[str, Any] | None = None,
    ) -> TranscriptPipelineResult:
        """Align diarization with transcription and build the transcript document."""
        LOGGER.debug(
            "Running transcript pipeline with %s ASR segments.",
            len(transcription.segments),
        )
        options: dict[str, Any] = dict(self._alignment_options)
        if alignment_overrides:
            options.update(dict(alignment_overrides))

        alignment_kwargs: dict[str, Any] = {}
        if "max_time_diff_seconds" in options:
            try:
                alignment_kwargs["max_time_diff_seconds"] = float(options["max_time_diff_seconds"])
            except (TypeError, ValueError):
                LOGGER.debug("Ignoring invalid alignment option 'max_time_diff_seconds'.")
        if "prefer_longer_segments" in options:
            alignment_kwargs["prefer_longer_segments"] = bool(options["prefer_longer_segments"])

        alignment = align_transcription_to_diarization(
            transcription,
            diarization,
            **alignment_kwargs,
        )
        metadata = self._build_metadata(transcription, diarization, alignment, episode_metadata)

        document, corrections = self.builder.build_document_from_alignment(
            alignment,
            metadata=metadata,
            auto_correct_names=auto_correct_names,
        )
        LOGGER.debug(
            "Transcript pipeline produced %s segments with %s corrections.",
            len(document.segments),
            len(corrections),
        )
        return TranscriptPipelineResult(
            alignment=alignment,
            document=document,
            corrections=corrections,
        )

    @staticmethod
    def _build_metadata(
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
        alignment: AlignmentResult,
        episode_metadata: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Compose metadata for the transcript output."""
        payload: dict[str, Any] = {}
        if episode_metadata:
            payload.update(dict(episode_metadata))

        payload.setdefault("transcription", asdict(transcription.metadata))
        payload.setdefault("diarization", diarization.metadata.to_dict())
        payload.setdefault("alignment_summary", alignment.metadata.to_dict())
        payload.setdefault("speaker_order", alignment.metadata.speakers)
        payload.setdefault("segment_count", len(alignment.segments))
        return payload
