"""Tests for the transcript builder integration."""

from __future__ import annotations

from types import SimpleNamespace

from show_scribe.pipelines.transcript.builder import TranscriptBuilder
from show_scribe.utils.name_correction import NameCorrector


def _build_corrector() -> NameCorrector:
    show_config = {
        "auto_correct_names": True,
        "name_correction": {
            "enabled": True,
            "auto_correct_threshold": 0.75,
            "fuzzy_match_cutoff": 0.7,
        },
        "cast_members": [
            {
                "canonical_name": "Michael Scott",
                "common_misspellings": ["Micheal Scott"],
            },
            {
                "canonical_name": "Dwight Schrute",
                "common_misspellings": ["Dwight Shrute"],
            },
        ],
    }

    voice_bank = SimpleNamespace(
        list_speakers=lambda: [],
        log_name_correction=lambda **_: None,
    )
    return NameCorrector(show_config, voice_bank=voice_bank)


def test_builder_applies_name_corrections() -> None:
    corrector = _build_corrector()
    builder = TranscriptBuilder(name_corrector=corrector)

    segments = [
        {"start": 15.2, "text": "Good morning!", "speaker_name": "Micheal Scott"},
        {"start": 20.0, "text": "Fact.", "speaker": "Dwight Shrute"},
    ]

    transcript_text, corrections = builder.build_text_transcript(segments)

    assert "[00:00:15]" in transcript_text
    assert "Michael Scott" in transcript_text
    assert "Dwight Schrute" in transcript_text

    assert len(corrections) == 2
    assert segments[0]["speaker_name"] == "Micheal Scott"

    corrected_segments, _ = builder.apply_name_corrections_to_segments(segments, mutate=False)
    assert corrected_segments[0]["speaker_name"] == "Michael Scott"
    assert corrected_segments[1]["speaker_name"] == "Dwight Schrute"
    assert corrected_segments[1]["original_speaker"] == "Dwight Shrute"
