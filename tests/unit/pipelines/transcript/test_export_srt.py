from __future__ import annotations

from src.show_scribe.pipelines.transcript.builder import TranscriptDocument, TranscriptSegment
from src.show_scribe.pipelines.transcript.export_srt import (
    render_srt,
    render_srt_with_alignment,
)


def _document() -> TranscriptDocument:
    segment = TranscriptSegment(
        segment_id=0,
        start=0.0,
        end=2.0,
        speaker="SPEAKER_00",
        text="sample line",
        speaker_confidence=0.75,
        words=[{"word": "sample", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        metadata={"asr_confidence": 0.8, "speaker_distribution": {"SPEAKER_00": 2.0}},
    )
    return TranscriptDocument(segments=[segment])


def test_render_srt_default_compact() -> None:
    doc = _document()
    output = render_srt(doc)
    assert "speaker_conf=" not in output


def test_render_srt_with_alignment_details() -> None:
    doc = _document()
    output = render_srt_with_alignment(doc)
    assert "speaker_conf=" in output
