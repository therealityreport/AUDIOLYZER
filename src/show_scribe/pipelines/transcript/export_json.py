"""Export transcript documents as structured JSON payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...utils.name_correction import NameCorrectionResult
from ..alignment.align_asr_diar import AlignmentResult
from .builder import TranscriptDocument

__all__ = ["build_transcript_payload"]


def build_transcript_payload(
    document: TranscriptDocument,
    *,
    alignment: AlignmentResult | None = None,
    corrections: Sequence[NameCorrectionResult] | None = None,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a serialisable payload representing the transcript outputs."""
    payload = document.to_dict()
    metadata = dict(payload.get("metadata", {}))
    if extras:
        metadata.update(dict(extras))
    payload["metadata"] = metadata

    if alignment is not None:
        alignment_payload = alignment.to_dict()
        payload["alignment"] = alignment_payload
        metadata.setdefault("alignment_summary", alignment_payload.get("metadata", {}))
        metadata.setdefault("speaker_order", alignment.metadata.speakers)

    if corrections:
        payload["name_corrections"] = [result.to_dict() for result in corrections]

    return payload
