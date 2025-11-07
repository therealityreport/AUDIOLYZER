"""Tests for the name correction utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from show_scribe.utils.name_correction import NameCorrector


class StubVoiceBank:
    """Lightweight stub for the VoiceBankManager interface."""

    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def list_speakers(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                key="michaelscott",
                display_name="Michael Scott",
                common_aliases=("Mike",),
                common_misspellings=("Micheal Scott",),
                phonetic_spelling=None,
                first_appearance_episode=None,
                total_segments=0,
                notes=None,
            )
        ]

    def log_name_correction(
        self,
        *,
        original: str,
        corrected: str,
        method: str,
        confidence: float,
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


@pytest.fixture()
def show_config() -> dict[str, object]:
    return {
        "auto_correct_names": True,
        "confidence_threshold": 0.75,
        "name_correction": {
            "enabled": True,
            "auto_correct_threshold": 0.75,
            "fuzzy_match_cutoff": 0.7,
            "preserve_formatting": False,
        },
        "cast_members": [
            {
                "canonical_name": "Michael Scott",
                "aliases": ["Michael"],
                "common_misspellings": ["Micheal Scott"],
                "role": "main",
            },
            {
                "canonical_name": "Dwight Schrute",
                "aliases": [],
                "common_misspellings": ["Dwight Shrute"],
            },
        ],
    }


def test_exact_canonical_match(show_config: dict[str, object]) -> None:
    corrector = NameCorrector(show_config, voice_bank=None)
    result = corrector.correct_name("Michael Scott")

    assert result.corrected == "Michael Scott"
    assert result.method == "canonical"
    assert not result.changed


def test_alias_match_logs_correction(show_config: dict[str, object]) -> None:
    voice_bank = StubVoiceBank()
    corrector = NameCorrector(show_config, voice_bank=voice_bank)

    result = corrector.correct_name("Mike", log_change=True)

    assert result.corrected == "Michael Scott"
    assert result.method == "alias"
    assert corrector.should_auto_correct(result)
    assert voice_bank.logged, "Expected correction to be logged to the voice bank."
    payload = voice_bank.logged[0]
    assert payload["original"] == "Mike"
    assert payload["corrected"] == "Michael Scott"
    assert payload["method"] == "alias"


def test_common_misspelling_corrected(show_config: dict[str, object]) -> None:
    corrector = NameCorrector(show_config, voice_bank=None)

    result = corrector.correct_name("Micheal Scott")

    assert result.corrected == "Michael Scott"
    assert result.method == "misspelling"
    assert corrector.should_auto_correct(result)


def test_fuzzy_match_provides_correction(show_config: dict[str, object]) -> None:
    corrector = NameCorrector(show_config, voice_bank=None)

    result = corrector.correct_name("Michel Scott")

    assert result.method in {"fuzzy", "canonical"}
    assert result.corrected == "Michael Scott"
    assert corrector.should_auto_correct(result)


def test_transcript_correction(show_config: dict[str, object]) -> None:
    corrector = NameCorrector(show_config, voice_bank=None)

    transcript = "[00:00:15] Micheal Scott: Hello!\n[00:00:20] Dwight Shrute: False."
    corrected = corrector.correct_transcript(transcript)

    assert "Michael Scott" in corrected
    assert "Dwight Schrute" in corrected
