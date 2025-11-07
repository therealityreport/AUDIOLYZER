"""Tests for ASR and diarization alignment."""

from __future__ import annotations

from src.show_scribe.pipelines.alignment.align_asr_diar import (
    AlignmentResult,
    align_transcription_to_diarization,
)
from src.show_scribe.pipelines.asr.whisper_local import (
    SegmentTranscription,
    TranscriptionMetadata,
    TranscriptionResult,
    WordTiming,
)
from src.show_scribe.pipelines.diarization.pyannote_pipeline import (
    DiarizationMetadata,
    DiarizationResult,
    DiarizationSegment,
)


def _build_transcription() -> TranscriptionResult:
    segments = [
        SegmentTranscription(
            segment_id=0,
            start=0.0,
            end=2.0,
            text="hello there",
            confidence=0.92,
            words=[
                WordTiming(word="hello", start=0.0, end=0.8, probability=0.9),
                WordTiming(word="there", start=1.0, end=1.9, probability=0.85),
            ],
        ),
        SegmentTranscription(
            segment_id=1,
            start=2.1,
            end=4.0,
            text="general kenobi",
            confidence=0.88,
            words=[
                WordTiming(word="general", start=2.1, end=3.0, probability=0.82),
                WordTiming(word="kenobi", start=3.0, end=3.9, probability=0.78),
            ],
        ),
    ]
    metadata = TranscriptionMetadata(language="en", duration=4.0)
    return TranscriptionResult(segments=segments, metadata=metadata)


def _build_diarization() -> DiarizationResult:
    segments = [
        DiarizationSegment(segment_id=0, start=0.0, end=1.5, speaker="SPEAKER_00", confidence=0.7),
        DiarizationSegment(segment_id=1, start=1.5, end=4.2, speaker="SPEAKER_01", confidence=0.6),
    ]
    metadata = DiarizationMetadata(
        model="pyannote/test",
        speaker_count=2,
        duration=4.2,
        inference_seconds=1.2,
        parameters={},
    )
    return DiarizationResult(segments=segments, metadata=metadata)


def test_alignment_assigns_speakers() -> None:
    transcription = _build_transcription()
    diarization = _build_diarization()

    result: AlignmentResult = align_transcription_to_diarization(transcription, diarization)

    assert len(result.segments) == 2
    assert result.metadata.words_aligned_count == 4
    assert result.metadata.unaligned_word_count == 0
    assert result.metadata.speakers == ["SPEAKER_00", "SPEAKER_01"]

    first, second = result.segments
    assert first.speaker == "SPEAKER_00"
    assert second.speaker == "SPEAKER_01"
    assert first.metadata["speaker_distribution"]["SPEAKER_00"] > 0.0
    assert second.metadata["speaker_distribution"]["SPEAKER_01"] > 0.0

    payload = result.as_transcript_segments()
    assert payload[0]["speaker"] == "SPEAKER_00"
    assert payload[0]["metadata"]["source"] == "alignment"
    assert payload[0]["metadata"]["aligned_words"] == len(first.words)


def test_alignment_handles_empty_diarization() -> None:
    transcription = _build_transcription()
    diarization = DiarizationResult(segments=[], metadata=_build_diarization().metadata)

    result = align_transcription_to_diarization(transcription, diarization)

    assert len(result.segments) == 2
    assert all(segment.speaker == "Unknown" for segment in result.segments)
    assert result.metadata.diarization_segment_count == 0
    assert result.metadata.words_aligned_count == 4


def test_alignment_synthesises_single_word_segments() -> None:
    segments = [
        SegmentTranscription(
            segment_id=0,
            start=0.0,
            end=1.0,
            text="hello",
            confidence=0.9,
            words=[],
        )
    ]
    transcription = TranscriptionResult(
        segments=segments,
        metadata=TranscriptionMetadata(language="en", duration=1.0),
    )
    diarization = _build_diarization()

    result = align_transcription_to_diarization(transcription, diarization)

    assert len(result.segments) == 1
    aligned_segment = result.segments[0]
    assert aligned_segment.words, "Synthetic word timing should be created."
    assert aligned_segment.words[0].word == "hello"
