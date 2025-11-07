"""Alignment of ASR transcripts with diarization segments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...utils.logging import get_logger
from ..asr.whisper_local import SegmentTranscription, TranscriptionResult, WordTiming
from ..diarization.pyannote_pipeline import DiarizationResult, DiarizationSegment

LOGGER = get_logger(__name__)

__all__ = [
    "AlignedSegment",
    "AlignedWord",
    "AlignmentMetadata",
    "AlignmentResult",
    "align_transcription_to_diarization",
]


@dataclass(slots=True)
class AlignedWord:
    """Word-level alignment data."""

    word: str
    start: float
    end: float
    speaker: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the word information."""
        payload: dict[str, Any] = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
        }
        if self.probability is not None:
            payload["probability"] = self.probability
        return payload


@dataclass(slots=True)
class AlignedSegment:
    """A transcription segment annotated with diarization metadata."""

    segment_id: int
    start: float
    end: float
    speaker: str
    text: str
    words: list[AlignedWord] = field(default_factory=list)
    speaker_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the segment."""
        payload: dict[str, Any] = {
            "id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }
        if self.speaker_confidence is not None:
            payload["speaker_confidence"] = self.speaker_confidence
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class AlignmentMetadata:
    """Aggregate details about an alignment run."""

    asr_segment_count: int
    diarization_segment_count: int
    words_aligned_count: int
    unaligned_word_count: int
    speakers: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable version."""
        return {
            "asr_segment_count": self.asr_segment_count,
            "diarization_segment_count": self.diarization_segment_count,
            "words_aligned_count": self.words_aligned_count,
            "unaligned_word_count": self.unaligned_word_count,
            "speakers": list(self.speakers),
        }


@dataclass(slots=True)
class AlignmentResult:
    """Alignment output suitable for transcript building."""

    segments: list[AlignedSegment]
    metadata: AlignmentMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialise the alignment result."""
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata.to_dict(),
        }

    def as_transcript_segments(self) -> list[dict[str, Any]]:
        """Return list-of-mapping representation for the transcript builder."""
        payload: list[dict[str, Any]] = []
        for segment in self.segments:
            segment_data = segment.to_dict()
            segment_data.setdefault("metadata", {}).update(
                {
                    "source": "alignment",
                    "aligned_words": len(segment.words),
                }
            )
            segment_data["speaker_name"] = segment_data.get("speaker")
            payload.append(segment_data)
        return payload


def align_transcription_to_diarization(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
    *,
    max_time_diff_seconds: float | None = None,
    prefer_longer_segments: bool = True,
) -> AlignmentResult:
    """Return aligned segments by matching ASR content with diarization speakers."""

    diar_segments = sorted(diarization.segments, key=lambda segment: (segment.start, segment.end))
    asr_segments = transcription.segments

    if not asr_segments:
        metadata = AlignmentMetadata(
            asr_segment_count=0,
            diarization_segment_count=len(diar_segments),
            words_aligned_count=0,
            unaligned_word_count=0,
            speakers=[],
        )
        return AlignmentResult(segments=[], metadata=metadata)

    diar_index = 0
    diar_count = len(diar_segments)

    def lookup_speaker(timepoint: float) -> DiarizationSegment | None:
        nonlocal diar_index
        if diar_count == 0:
            return None

        if diar_index >= diar_count:
            diar_index = diar_count - 1

        while diar_index < diar_count and diar_segments[diar_index].end <= timepoint:
            diar_index += 1
        if diar_index < diar_count:
            candidate = diar_segments[diar_index]
            if candidate.start <= timepoint < candidate.end:
                return candidate

        if diar_index > 0:
            previous = diar_segments[diar_index - 1]
            if previous.start <= timepoint < previous.end:
                diar_index -= 1
                return previous

        if timepoint < diar_segments[0].start:
            diar_index = 0
            return diar_segments[0]

        diar_index = diar_count - 1
        return diar_segments[-1]

    aligned_segments: list[AlignedSegment] = []
    speakers_seen: list[str] = []
    words_aligned = 0
    unaligned_words = 0

    for segment in asr_segments:
        aligned_words, newly_unaligned = _align_words(
            segment,
            lookup_speaker,
            max_time_diff=max_time_diff_seconds,
        )
        if not aligned_words:
            LOGGER.debug("Segment %s produced no aligned words; skipping.", segment.segment_id)
            continue

        words_aligned += len(aligned_words)
        unaligned_words += newly_unaligned

        speaker_distribution: dict[str, float] = {}
        speaker_counts: dict[str, int] = {}
        speaker_confidence_map: dict[str, float] = {}

        for word in aligned_words:
            if word.speaker == "Unknown":
                continue
            duration = max(word.end - word.start, 0.0)
            speaker_distribution[word.speaker] = (
                speaker_distribution.get(word.speaker, 0.0) + duration
            )
            speaker_counts[word.speaker] = speaker_counts.get(word.speaker, 0) + 1
            if word.probability is not None:
                speaker_confidence_map[word.speaker] = word.probability

        segment_speaker, speaker_confidence = _select_speaker(
            segment,
            speaker_distribution,
            speaker_counts,
            prefer_longer_segments=prefer_longer_segments,
        )
        if segment_speaker not in speaker_confidence_map and speaker_confidence is None:
            speaker_confidence = speaker_confidence_map.get(segment_speaker)

        words_for_segment = [
            AlignedWord(
                word=word.word,
                start=word.start,
                end=word.end,
                speaker=word.speaker,
                probability=word.probability,
            )
            for word in aligned_words
        ]

        distribution_copy = {
            speaker: round(duration, 6) for speaker, duration in speaker_distribution.items()
        }

        aligned_segment = AlignedSegment(
            segment_id=segment.segment_id,
            start=float(segment.start),
            end=float(segment.end),
            speaker=segment_speaker,
            text=segment.text,
            words=words_for_segment,
            speaker_confidence=speaker_confidence,
            metadata={
                "asr_confidence": segment.confidence,
                "speaker_distribution": distribution_copy,
                "speaker_counts": dict(speaker_counts),
            },
        )
        aligned_segments.append(aligned_segment)

        for speaker in speaker_counts:
            if speaker not in speakers_seen:
                speakers_seen.append(speaker)
        if segment_speaker != "Unknown" and segment_speaker not in speakers_seen:
            speakers_seen.append(segment_speaker)

    metadata = AlignmentMetadata(
        asr_segment_count=len(asr_segments),
        diarization_segment_count=len(diar_segments),
        words_aligned_count=words_aligned,
        unaligned_word_count=unaligned_words,
        speakers=speakers_seen,
    )
    return AlignmentResult(segments=aligned_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_words(
    segment: SegmentTranscription,
    lookup: Callable[[float], DiarizationSegment | None],
    *,
    max_time_diff: float | None = None,
) -> tuple[list[AlignedWord], int]:
    """Return aligned words for a single ASR segment."""

    words_source: Sequence[WordTiming] = segment.words
    if not words_source:
        synthetic_word = WordTiming(
            word=segment.text,
            start=float(segment.start),
            end=float(segment.end),
            probability=segment.confidence,
        )
        words_source = (synthetic_word,)

    aligned: list[AlignedWord] = []
    unaligned = 0

    max_threshold: float | None
    if max_time_diff is None:
        max_threshold = None
    else:
        try:
            max_threshold = max(0.0, float(max_time_diff))
        except (TypeError, ValueError):
            max_threshold = None

    for word in words_source:
        start = _safe_float(word.start, default=float(segment.start))
        end = _safe_float(word.end, default=start)
        if end < start:
            end = start

        midpoint = (start + end) / 2.0
        diar_segment = lookup(midpoint)
        speaker = "Unknown"
        diar_confidence = None

        if diar_segment is not None:
            outside_bounds = False
            if max_threshold is not None:
                outside_bounds = not (
                    (diar_segment.start - max_threshold)
                    <= midpoint
                    <= (diar_segment.end + max_threshold)
                )
            else:
                outside_bounds = not (diar_segment.start <= midpoint <= diar_segment.end)

            if not outside_bounds:
                speaker = diar_segment.speaker or "Unknown"
                diar_confidence = diar_segment.confidence

        probability = _safe_probability(getattr(word, "probability", None))
        if probability is None and diar_confidence is not None:
            probability = diar_confidence

        aligned.append(
            AlignedWord(
                word=word.word,
                start=start,
                end=end,
                speaker=speaker,
                probability=probability,
            )
        )
        if speaker == "Unknown":
            unaligned += 1
    return aligned, unaligned


def _select_speaker(
    segment: SegmentTranscription,
    distribution: Mapping[str, float],
    counts: Mapping[str, int],
    *,
    prefer_longer_segments: bool,
) -> tuple[str, float | None]:
    """Return the dominant speaker for a segment and confidence."""

    if prefer_longer_segments and distribution:
        total = sum(distribution.values())
        speaker, coverage = max(distribution.items(), key=lambda item: item[1])
        confidence = coverage / total if total > 0 else None
        return speaker, confidence

    if counts:
        speaker = max(counts.items(), key=lambda item: item[1])[0]
        return speaker, segment.confidence

    if distribution:
        total = sum(distribution.values())
        speaker, coverage = max(distribution.items(), key=lambda item: item[1])
        confidence = coverage / total if total > 0 else None
        return speaker, confidence

    return "Unknown", segment.confidence


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
