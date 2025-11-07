"""Helpers for recording ASR provider spend to on-disk logs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_COST_LOG = Path("reports/asr_costs.jsonl")
DEFAULT_COST_SUMMARY = Path("reports/asr_costs_summary.json")

__all__ = ["ASRCostEvent", "record_asr_cost"]


@dataclass(slots=True)
class ASRCostEvent:
    """Single cost observation for an ASR transcription run."""

    provider: str
    model: str
    cost_usd: float
    duration_seconds: float | None = None
    media_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Normalise floats for readability
        payload["cost_usd"] = round(float(self.cost_usd), 4)
        if payload.get("duration_seconds") is not None:
            payload["duration_seconds"] = round(float(self.duration_seconds or 0.0), 4)
        return payload


def record_asr_cost(
    *,
    provider: str,
    model: str,
    cost_usd: float | None,
    duration_seconds: float | None = None,
    media_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    log_path: Path | None = None,
    summary_path: Path | None = None,
) -> None:
    """Persist an ASR cost data point and update roll-up metrics."""

    if cost_usd is None:
        LOGGER.debug("No cost information supplied for provider %s; skipping record.", provider)
        return

    try:
        cost_value = float(cost_usd)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid cost value %s for provider %s; skipping record.",
            cost_usd,
            provider,
        )
        return

    duration_value: float | None = None
    if duration_seconds is not None:
        try:
            duration_value = float(duration_seconds)
        except (TypeError, ValueError):
            duration_value = None

    event = ASRCostEvent(
        provider=provider,
        model=model,
        cost_usd=cost_value,
        duration_seconds=duration_value,
        media_path=str(media_path) if media_path is not None else None,
        metadata=dict(metadata) if metadata else {},
    )

    target_log = log_path or DEFAULT_COST_LOG
    target_summary = summary_path or DEFAULT_COST_SUMMARY
    target_log.parent.mkdir(parents=True, exist_ok=True)

    with target_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False))
        handle.write("\n")

    _update_summary(target_summary, event)


def _update_summary(summary_path: Path, event: ASRCostEvent) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except json.JSONDecodeError:
            LOGGER.warning("Unable to parse existing ASR cost summary; starting fresh.")
            summary = {}
    else:
        summary = {}

    total_cost = float(summary.get("total_cost_usd", 0.0)) + event.cost_usd
    total_runs = int(summary.get("transcription_count", 0)) + 1

    providers: dict[str, Any] = summary.get("providers", {}) or {}
    provider_summary = providers.setdefault(
        event.provider,
        {"total_cost_usd": 0.0, "transcription_count": 0, "models": {}},
    )
    provider_summary["total_cost_usd"] = round(
        float(provider_summary.get("total_cost_usd", 0.0)) + event.cost_usd,
        4,
    )
    provider_summary["transcription_count"] = (
        int(provider_summary.get("transcription_count", 0)) + 1
    )

    model_summary = provider_summary["models"].setdefault(
        event.model,
        {"total_cost_usd": 0.0, "transcription_count": 0},
    )
    model_summary["total_cost_usd"] = round(
        float(model_summary.get("total_cost_usd", 0.0)) + event.cost_usd,
        4,
    )
    model_summary["transcription_count"] = int(model_summary.get("transcription_count", 0)) + 1

    provider_summary["last_used"] = event.timestamp
    provider_summary["last_media_path"] = event.media_path

    summary.update(
        {
            "total_cost_usd": round(total_cost, 4),
            "transcription_count": total_runs,
            "providers": providers,
            "last_updated": event.timestamp,
        }
    )

    if event.duration_seconds is not None:
        cumulative_duration = (
            float(summary.get("total_duration_seconds", 0.0)) + event.duration_seconds
        )
        summary["total_duration_seconds"] = round(cumulative_duration, 2)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
