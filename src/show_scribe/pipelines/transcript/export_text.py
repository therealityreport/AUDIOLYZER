"""Export transcript documents as formatted plain text."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .builder import TranscriptBuilder, TranscriptDocument, TranscriptSegment

__all__ = ["render_plain_text", "render_plain_text_with_alignment"]


def render_plain_text(
    document: TranscriptDocument,
    *,
    header: Mapping[str, object] | None = None,
    include_metadata: bool = True,
    timestamp_formatter: Callable[[float], str] | None = None,
    include_alignment_details: bool = False,
) -> str:
    """Return a human-readable plain text transcript."""
    formatter = timestamp_formatter or TranscriptBuilder._format_timestamp
    lines: list[str] = []

    if include_metadata:
        _append_metadata(lines, document.metadata, header)

    for segment in document.segments:
        timestamp = formatter(segment.start)
        speaker = segment.speaker or "Unknown"
        text = segment.text or ""
        lines.append(f"[{timestamp}] {speaker}: {text}")

        if not include_alignment_details:
            continue

        detail_line = _summarise_alignment(segment)
        if detail_line:
            lines.append(f"    {detail_line}")

        words_line = _format_word_timeline(segment.words, formatter)
        if words_line:
            lines.append(f"    {words_line}")

    return "\n".join(lines).rstrip()


def render_plain_text_with_alignment(
    document: TranscriptDocument,
    *,
    header: Mapping[str, object] | None = None,
    include_metadata: bool = True,
    timestamp_formatter: Callable[[float], str] | None = None,
) -> str:
    """Convenience wrapper that includes alignment details by default."""
    return render_plain_text(
        document,
        header=header,
        include_metadata=include_metadata,
        timestamp_formatter=timestamp_formatter,
        include_alignment_details=True,
    )


def _append_metadata(
    lines: list[str],
    document_metadata: Mapping[str, object],
    header: Mapping[str, object] | None,
) -> None:
    """Append header metadata in a consistent order."""
    combined: list[tuple[str, object]] = []

    if header:
        combined.extend((str(key), header[key]) for key in header)

    for key, value in document_metadata.items():
        key_str = str(key)
        if header and key_str in header:
            continue
        combined.append((key_str, value))

    if not combined:
        return

    for key, value in combined:
        normalised_key = key.replace("_", " ").strip().title()
        lines.append(f"{normalised_key}: {value}")

    lines.append("")


def _summarise_alignment(segment: TranscriptSegment) -> str:
    """Return a short summary of alignment metadata for a segment."""
    details: list[str] = []

    if segment.speaker_confidence is not None:
        details.append(f"speaker_conf={segment.speaker_confidence:.2f}")

    asr_conf = segment.metadata.get("asr_confidence")
    if isinstance(asr_conf, (float, int)):
        details.append(f"asr_conf={float(asr_conf):.2f}")

    distribution = segment.metadata.get("speaker_distribution")
    if isinstance(distribution, Mapping) and distribution:
        distribution_parts = [
            f"{speaker}:{float(duration):.2f}s"
            for speaker, duration in sorted(distribution.items())
        ]
        details.append("mix=" + ", ".join(distribution_parts))

    speaker_counts = segment.metadata.get("speaker_counts")
    if isinstance(speaker_counts, Mapping) and speaker_counts:
        counts_formatted = ", ".join(
            f"{speaker}:{int(count)}" for speaker, count in sorted(speaker_counts.items())
        )
        details.append("speaker_counts=" + counts_formatted)

    if segment.words:
        details.append(f"words={len(segment.words)}")

    return "; ".join(details)


def _format_word_timeline(
    words: Sequence[Mapping[str, Any]],
    formatter: Callable[[float], str],
) -> str:
    """Render a compact per-word timeline for aligned segments."""
    if not words:
        return ""

    formatted: list[str] = []
    for word in words[:12]:
        label = str(word.get("word", "")).strip()
        if not label:
            continue
        start = formatter(float(word.get("start", 0.0)))
        end = formatter(float(word.get("end", 0.0)))
        speaker = word.get("speaker")
        if speaker:
            speaker = str(speaker)
        else:
            speaker = None
        fragment = f"{label}[{start}-{end}]"
        if speaker and speaker != "Unknown":
            fragment += f"@{speaker}"
        formatted.append(fragment)

    if len(words) > 12:
        formatted.append(f"...(+{len(words) - 12} more)")

    return "Words: " + ", ".join(formatted)
