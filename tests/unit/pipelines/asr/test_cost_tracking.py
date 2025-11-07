from __future__ import annotations

import json

from src.show_scribe.pipelines.asr.cost_tracking import record_asr_cost


def test_record_asr_cost_persists_event(tmp_path) -> None:
    log_path = tmp_path / "costs.jsonl"
    summary_path = tmp_path / "summary.json"

    record_asr_cost(
        provider="whisper_api",
        model="whisper-1",
        cost_usd=1.2345,
        duration_seconds=123.456,
        media_path=str(tmp_path / "example.wav"),
        metadata={"episode_id": "S05E01"},
        log_path=log_path,
        summary_path=summary_path,
    )

    assert log_path.exists()
    event_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert event_lines, "Expected at least one logged cost event"
    event_payload = json.loads(event_lines[-1])
    assert event_payload["provider"] == "whisper_api"
    assert event_payload["model"] == "whisper-1"
    assert event_payload["cost_usd"] == round(1.2345, 4)
    assert event_payload["metadata"]["episode_id"] == "S05E01"

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["total_cost_usd"] == round(1.2345, 4)
    assert summary_payload["transcription_count"] == 1
    provider_summary = summary_payload["providers"]["whisper_api"]
    assert provider_summary["transcription_count"] == 1
    assert provider_summary["models"]["whisper-1"]["total_cost_usd"] == round(1.2345, 4)
