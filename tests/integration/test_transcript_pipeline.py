"""Integration test covering ASR, diarization, and alignment orchestration."""

from __future__ import annotations

from src.show_scribe.pipelines.alignment.align_asr_diar import AlignmentResult
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
from src.show_scribe.pipelines.transcript.export_json import build_transcript_payload
from src.show_scribe.pipelines.transcript.export_srt import render_srt
from src.show_scribe.pipelines.transcript.export_text import render_plain_text
from src.show_scribe.pipelines.transcript.pipeline import TranscriptPipeline


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


def test_transcript_pipeline_runs_end_to_end() -> None:
    transcription = _build_transcription()
    diarization = _build_diarization()
    pipeline = TranscriptPipeline()

    result = pipeline.run(
        transcription,
        diarization,
        episode_metadata={"episode_id": "S01E01"},
    )

    assert isinstance(result.alignment, AlignmentResult)
    assert len(result.document.segments) == 2
    assert result.document.metadata["segment_count"] == 2
    assert result.document.metadata["alignment_summary"]["asr_segment_count"] == 2

    text_output = render_plain_text(result.document)
    assert "SPEAKER_00" in text_output
    assert "[00:00:00.000]" in text_output

    srt_output = render_srt(result.document)
    assert "SPEAKER_00" in srt_output
    assert "00:00:00,000 --> 00:00:02,000" in srt_output

    json_payload = build_transcript_payload(
        result.document,
        alignment=result.alignment,
        corrections=result.corrections,
    )
    assert json_payload["metadata"]["segment_count"] == 2
    assert json_payload["alignment"]["metadata"]["asr_segment_count"] == 2
    assert json_payload.get("name_corrections", []) == []
