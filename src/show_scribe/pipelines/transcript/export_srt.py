"""Export transcript documents in SubRip (SRT) subtitle format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .builder import TranscriptDocument, TranscriptSegment

__all__ = ["render_srt", "render_srt_with_alignment"]


def render_srt(document: TranscriptDocument, *, include_alignment_details: bool = False) -> str:
    """Return a SubRip formatted transcript."""
    entries: list[str] = []
    for index, segment in enumerate(document.segments, start=1):
        start_ts = _format_srt_timestamp(segment.start)
        end_ts = _format_srt_timestamp(segment.end)
        speaker = segment.speaker or "Unknown"
        text_lines = [f"[{speaker}] {segment.text or ''}"]

        if include_alignment_details:
            detail_line = _summarise_alignment(segment)
            if detail_line:
                text_lines.append(detail_line)
            words_line = _summarise_words(segment.words)
            if words_line:
                text_lines.append(words_line)

        entry = f"{index}\n{start_ts} --> {end_ts}\n" + "\n".join(text_lines)
        entries.append(entry.strip())

    return "\n\n".join(entries)


def render_srt_with_alignment(document: TranscriptDocument) -> str:
    """Return an SRT transcript that always includes alignment details."""
    return render_srt(document, include_alignment_details=True)


def _format_srt_timestamp(value: float) -> str:
    """Format seconds into ``HH:MM:SS,mmm`` for SRT rendering."""
    total_milliseconds = round(max(value, 0.0) * 1000)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _summarise_alignment(segment: TranscriptSegment) -> str:
    details: list[str] = []

    if segment.speaker_confidence is not None:
        details.append(f"speaker_conf={segment.speaker_confidence:.2f}")

    asr_conf = segment.metadata.get("asr_confidence")
    if isinstance(asr_conf, (float, int)):
        details.append(f"asr_conf={float(asr_conf):.2f}")

    distribution = segment.metadata.get("speaker_distribution")
    if isinstance(distribution, dict) and distribution:
        parts = [
            f"{name}:{float(duration):.2f}s" for name, duration in sorted(distribution.items())
        ]
        details.append("mix=" + ", ".join(parts))

    if segment.words:
        details.append(f"words={len(segment.words)}")

    return " | ".join(details)


def _summarise_words(words: Sequence[Mapping[str, Any]]) -> str:
    if not words:
        return ""

    first = words[0]
    last = words[-1]
    try:
        coverage = float(last.get("end", 0.0)) - float(first.get("start", 0.0))
    except (TypeError, ValueError):
        coverage = 0.0

    return f"word_window={max(coverage, 0.0):.2f}s"
