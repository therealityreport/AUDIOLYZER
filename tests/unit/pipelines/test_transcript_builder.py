"""Tests for the transcript builder integration with name correction."""

from __future__ import annotations

from dataclasses import dataclass

from src.show_scribe.pipelines.transcript.builder import TranscriptBuilder
from src.show_scribe.storage.voice_bank_manager import SpeakerProfile
from src.show_scribe.utils.name_correction import NameCorrector


@dataclass(slots=True)
class DummyVoiceBank:
    logged: list[dict[str, object]]

    def list_speakers(self) -> list[SpeakerProfile]:
        return [
            SpeakerProfile(
                id=1,
                key="dwightschrute",
                display_name="Dwight Schrute",
                common_aliases=("Dwight",),
                common_misspellings=("Dwite Schrute",),
                phonetic_spelling=None,
                first_appearance_episode=None,
                total_segments=0,
                notes=None,
            ),
        ]

    def log_name_correction(
        self,
        *,
        original: str,
        corrected: str,
        method: str,
        confidence: float | None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.logged.append(
            {
                "original": original,
                "corrected": corrected,
                "method": method,
                "confidence": confidence,
                "metadata": metadata or {},
            }
        )


def make_corrector() -> NameCorrector:
    config = {
        "show_name": "The Office",
        "cast_members": [
            {
                "canonical_name": "Dwight Schrute",
                "common_misspellings": ["Dwite Schrute"],
                "aliases": ["Dwight"],
                "role": "main",
            }
        ],
        "name_correction": {
            "enabled": True,
            "auto_correct_threshold": 0.85,
            "fuzzy_match_cutoff": 0.75,
        },
    }
    return NameCorrector(config, DummyVoiceBank(logged=[]))


def test_apply_name_corrections_updates_segments() -> None:
    corrector = make_corrector()
    builder = TranscriptBuilder(name_corrector=corrector)
    segments = [
        {"speaker": "Dwite Schrute", "text": "Fact: bears eat beets.", "start": 0.0},
        {"speaker": "Jim Halpert", "text": "Bears. Beets. Battlestar Galactica.", "start": 1.5},
    ]

    corrected_segments, corrections = builder.apply_name_corrections_to_segments(segments)

    assert corrected_segments[0]["speaker"] == "Dwight Schrute"
    assert corrected_segments[1]["speaker"] == "Jim Halpert"
    assert len(corrections) == 1


def test_build_text_transcript_applies_corrections() -> None:
    corrector = make_corrector()
    builder = TranscriptBuilder(name_corrector=corrector)
    segments = [
        {"speaker": "Dwite Schrute", "text": "Fact: bears eat beets.", "start": 2.75},
    ]

    transcript, corrections = builder.build_text_transcript(segments)

    assert transcript.startswith("[00:00:02.750] Dwight Schrute: Fact")
    assert corrections and corrections[0].corrected == "Dwight Schrute"


def test_build_text_transcript_without_corrector_passes_through() -> None:
    builder = TranscriptBuilder(name_corrector=None)
    segments = [
        {"speaker": "Unknown Speaker", "text": "Hello world", "start": 3.0},
    ]

    transcript, corrections = builder.build_text_transcript(segments)

    assert transcript.startswith("[00:00:03.000] Unknown Speaker: Hello world")
    assert corrections == []


def test_build_document_normalises_segments() -> None:
    builder = TranscriptBuilder(name_corrector=None)
    segments = [
        {
            "start": "1.0",
            "end": "2.5",
            "speaker_label": "SPEAKER_00",
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": "1.0", "end": "1.4", "probability": "0.6"},
                {"word": "world", "start": 1.5, "end": 2.4, "probability": 0.8},
            ],
        }
    ]

    document, corrections = builder.build_document(segments, metadata={"episode_id": "S01E01"})

    assert corrections == []
    assert document.metadata["segment_count"] == 1
    assert document.metadata["speaker_order"] == ["SPEAKER_00"]
    assert document.metadata["total_duration_seconds"] == document.segments[0].duration

    segment = document.segments[0]
    assert segment.segment_id == 0
    assert segment.start == 1.0
    assert segment.end == 2.5
    assert segment.speaker == "SPEAKER_00"
    assert segment.words[0]["probability"] == 0.6

    payload = document.to_dict()
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["metadata"]["episode_id"] == "S01E01"
