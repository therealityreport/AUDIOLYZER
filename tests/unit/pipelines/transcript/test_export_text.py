from __future__ import annotations

from src.show_scribe.pipelines.transcript.builder import TranscriptDocument, TranscriptSegment
from src.show_scribe.pipelines.transcript.export_text import (
    render_plain_text,
    render_plain_text_with_alignment,
)


def _build_document() -> TranscriptDocument:
    segment = TranscriptSegment(
        segment_id=0,
        start=0.0,
        end=1.5,
        speaker="SPEAKER_00",
        text="hello world",
        speaker_confidence=0.85,
        words=[
            {"word": "hello", "start": 0.0, "end": 0.7, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.7, "end": 1.4, "speaker": "SPEAKER_00"},
        ],
        metadata={"asr_confidence": 0.9, "speaker_distribution": {"SPEAKER_00": 1.5}},
    )
    return TranscriptDocument(segments=[segment])


def test_render_plain_text_default_is_compact() -> None:
    doc = _build_document()
    output = render_plain_text(doc)
    assert "Words:" not in output
    assert output.count("SPEAKER_00") == 1


def test_render_plain_text_with_alignment_contains_details() -> None:
    doc = _build_document()
    output = render_plain_text_with_alignment(doc)
    assert "Words:" in output
    assert "speaker_conf=" in output
