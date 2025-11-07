"""Transcript assembly helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...utils.name_correction import NameCorrectionResult, NameCorrector
from ..alignment.align_asr_diar import AlignmentResult

__all__ = [
    "TranscriptBuilder",
    "TranscriptDocument",
    "TranscriptSegment",
]


@dataclass(slots=True)
class TranscriptSegment:
    """Unified representation of a transcript segment."""

    segment_id: int
    start: float
    end: float
    speaker: str
    text: str
    speaker_confidence: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Return the segment duration in seconds."""
        return max(self.end - self.start, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the segment to a dictionary."""
        payload: dict[str, Any] = {
            "id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            "duration": self.duration,
        }
        if self.speaker_confidence is not None:
            payload["speaker_confidence"] = self.speaker_confidence
        if self.words:
            payload["words"] = [dict(word) for word in self.words]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class TranscriptDocument:
    """Container representing a fully assembled transcript."""

    segments: list[TranscriptSegment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the transcript document."""
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": dict(self.metadata),
        }

    @property
    def speakers(self) -> list[str]:
        """Return the list of speakers ordered by first appearance."""
        seen: set[str] = set()
        ordered: list[str] = []
        for segment in self.segments:
            if segment.speaker and segment.speaker not in seen:
                seen.add(segment.speaker)
                ordered.append(segment.speaker)
        return ordered

    @property
    def total_duration(self) -> float:
        """Total duration across all segments."""
        return sum(segment.duration for segment in self.segments)


class TranscriptBuilder:
    """Constructs transcripts and applies name corrections."""

    def __init__(self, *, name_corrector: NameCorrector | None = None) -> None:
        self.name_corrector = name_corrector

    # ------------------------------------------------------------------
    # Segment operations
    # ------------------------------------------------------------------
    def apply_name_corrections_to_segments(
        self,
        segments: Sequence[Mapping[str, Any]],
        *,
        mutate: bool = False,
        auto_only: bool = True,
    ) -> tuple[list[MutableMapping[str, Any]], list[NameCorrectionResult]]:
        """Return segments with corrected speaker labels and applied corrections."""
        if not segments:
            return [], []

        target_segments: list[MutableMapping[str, Any]]
        if mutate:
            target_segments = [
                segment if isinstance(segment, MutableMapping) else dict(segment)
                for segment in segments
            ]
        else:
            target_segments = [dict(segment) for segment in segments]

        if self.name_corrector is None or not self.name_corrector.enabled:
            return target_segments, []

        applied: list[NameCorrectionResult] = []
        for segment in target_segments:
            speaker_key = "speaker_name" if "speaker_name" in segment else "speaker"
            speaker = segment.get(speaker_key) or segment.get("speaker")
            if not isinstance(speaker, str) or not speaker.strip():
                continue

            result = self.name_corrector.correct_name(speaker, log_change=False)
            if auto_only and not self.name_corrector.should_auto_correct(result):
                continue
            if not result.changed:
                continue

            segment.setdefault("original_speaker", speaker)
            segment["speaker_name"] = result.corrected
            if speaker_key != "speaker_name":
                segment[speaker_key] = result.corrected
            applied.append(result)
            self.name_corrector.record_correction(result)

        return target_segments, applied

    # ------------------------------------------------------------------
    # Transcript assembly and rendering
    # ------------------------------------------------------------------
    def build_document(
        self,
        segments: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        auto_correct_names: bool = True,
    ) -> tuple[TranscriptDocument, list[NameCorrectionResult]]:
        """Return a structured transcript document and applied name corrections."""

        corrected_segments, corrections = self.apply_name_corrections_to_segments(
            segments,
            mutate=False,
            auto_only=auto_correct_names,
        )

        transcript_segments: list[TranscriptSegment] = [
            self._segment_from_mapping(idx, segment)
            for idx, segment in enumerate(corrected_segments)
        ]

        document = TranscriptDocument(segments=transcript_segments, metadata=dict(metadata or {}))
        document.metadata.setdefault("segment_count", len(transcript_segments))
        document.metadata.setdefault("speaker_order", document.speakers)
        document.metadata.setdefault("total_duration_seconds", document.total_duration)

        return document, corrections

    def build_document_from_alignment(
        self,
        alignment: AlignmentResult,
        *,
        metadata: Mapping[str, Any] | None = None,
        auto_correct_names: bool = True,
    ) -> tuple[TranscriptDocument, list[NameCorrectionResult]]:
        """Build a document using alignment output as the canonical segment source."""
        alignment_metadata = alignment.metadata.to_dict()
        payload_metadata: dict[str, Any] = {
            "alignment": alignment_metadata,
        }
        if metadata:
            payload_metadata.update(dict(metadata))

        transcript_segments = alignment.as_transcript_segments()
        document, corrections = self.build_document(
            transcript_segments,
            metadata=payload_metadata,
            auto_correct_names=auto_correct_names,
        )
        return document, corrections

    def build_text_transcript_from_alignment(
        self,
        alignment: AlignmentResult,
        *,
        include_timestamps: bool = True,
        timestamp_formatter: Callable[[float], str] | None = None,
        auto_correct_names: bool = True,
    ) -> tuple[str, list[NameCorrectionResult]]:
        """Render a text transcript using aligned segments as the source."""
        segments = alignment.as_transcript_segments()
        return self.build_text_transcript(
            segments,
            include_timestamps=include_timestamps,
            timestamp_formatter=timestamp_formatter,
            auto_correct_names=auto_correct_names,
        )

    def build_text_transcript(
        self,
        segments: Sequence[Mapping[str, Any]],
        *,
        include_timestamps: bool = True,
        timestamp_formatter: Callable[[float], str] | None = None,
        auto_correct_names: bool = True,
    ) -> tuple[str, list[NameCorrectionResult]]:
        """Return a text transcript alongside applied name corrections."""
        document, corrections = self.build_document(
            segments,
            metadata=None,
            auto_correct_names=auto_correct_names,
        )

        lines: list[str] = []
        for segment in document.segments:
            text = segment.text
            speaker = segment.speaker or "Unknown"

            if include_timestamps:
                formatter = timestamp_formatter or self._format_timestamp
                timestamp_value = formatter(segment.start)
                timestamp = str(timestamp_value)
                line = f"[{timestamp}] {speaker}: {text}"
                parts = timestamp.split(".", 1)
                if len(parts) == 2 and parts[1] != "000":
                    line = f"{line} [{parts[0]}]"
                lines.append(line)
                continue
            lines.append(f"{speaker}: {text}")

        transcript_text = "\n".join(lines)
        if (
            self.name_corrector is not None
            and self.name_corrector.enabled
            and not auto_correct_names
        ):
            corrected = self.name_corrector.correct_transcript(
                transcript_text,
                return_details=True,
            )
            if isinstance(corrected, tuple):
                transcript_text, post_corrections = corrected
            else:
                transcript_text = corrected
                post_corrections = []
            corrections.extend(post_corrections)
        return transcript_text, corrections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_timestamp(value: float) -> str:
        """Format seconds into HH:MM:SS.mmm."""
        total_milliseconds = round(max(value, 0.0) * 1000)
        total_seconds, milliseconds = divmod(total_milliseconds, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _segment_from_mapping(index: int, segment: Mapping[str, Any]) -> TranscriptSegment:
        """Convert a loosely-typed mapping into a :class:`TranscriptSegment`."""
        start_raw = segment.get("start", 0.0)
        end_raw = segment.get("end", start_raw)
        try:
            start = float(start_raw)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(end_raw)
        except (TypeError, ValueError):
            end = start

        speaker = (
            segment.get("speaker_name")
            or segment.get("speaker")
            or segment.get("speaker_label")
            or f"SPEAKER_{index:02d}"
        )
        text = str(segment.get("text", "")).strip()

        speaker_confidence = segment.get("speaker_confidence")
        if speaker_confidence is not None:
            try:
                speaker_confidence = float(speaker_confidence)
            except (TypeError, ValueError):
                speaker_confidence = None

        words_normalised: list[dict[str, Any]] = []

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        words_raw = segment.get("words") or []
        if isinstance(words_raw, Sequence) and not isinstance(words_raw, (str, bytes)):
            for entry in words_raw:
                if isinstance(entry, Mapping):
                    word_start = _safe_float(entry.get("start", start), start)
                    word_end = _safe_float(entry.get("end", word_start), word_start)
                    probability = entry.get("probability")
                    if probability is not None:
                        try:
                            probability = float(probability)
                        except (TypeError, ValueError):
                            probability = None
                    speaker_label = entry.get("speaker")
                    if speaker_label is not None and not isinstance(speaker_label, str):
                        speaker_label = str(speaker_label)
                    words_normalised.append(
                        {
                            "word": entry.get("word") or entry.get("token") or "",
                            "start": word_start,
                            "end": word_end,
                            "probability": probability,
                            "speaker": speaker_label,
                        }
                    )
                else:
                    words_normalised.append(
                        {
                            "word": str(entry),
                            "start": start,
                            "end": end,
                            "probability": None,
                            "speaker": None,
                        }
                    )

        metadata_raw = segment.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

        return TranscriptSegment(
            segment_id=index,
            start=start,
            end=end,
            speaker=str(speaker) if speaker is not None else f"SPEAKER_{index:02d}",
            text=text,
            speaker_confidence=speaker_confidence,
            words=words_normalised,
            metadata=metadata,
        )
