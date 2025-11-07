"""Cast name normalization and correction utilities."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.voice_bank_manager import VoiceBankManager

__all__ = ["NameCorrectionResult", "NameCorrector"]

LOGGER = get_logger(__name__)

SPEAKER_PATTERN = re.compile(
    r"^(?P<prefix>\s*(?:\[[^\]\n]+\]\s*)?)(?P<name>[^:\n]+?)(?P<separator>:\s)",
    re.MULTILINE,
)


@dataclass(slots=True)
class NameCorrectionResult:
    """Structured response describing the outcome of a correction attempt."""

    original_input: str
    cleaned_input: str
    corrected: str
    canonical: str | None
    method: str
    confidence: float
    metadata: dict[str, Any]

    @property
    def changed(self) -> bool:
        return self.corrected != self.cleaned_input

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the result."""
        return {
            "original": self.cleaned_input,
            "corrected": self.corrected,
            "canonical": self.canonical,
            "method": self.method,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class NameCorrector:
    """Ensures consistent and correct spelling of cast member names."""

    def __init__(
        self,
        show_config: Mapping[str, Any],
        voice_bank: VoiceBankManager | None = None,
        *,
        log_corrections: bool = True,
    ) -> None:
        self.show_config = show_config
        self.voice_bank = voice_bank
        self._log_enabled = log_corrections

        options = show_config.get("name_correction", {}) or {}
        self._enabled = bool(options.get("enabled", True))
        self._auto_threshold = float(options.get("auto_correct_threshold", 0.9))
        self._fuzzy_cutoff = float(options.get("fuzzy_match_cutoff", 0.8))
        self._case_sensitive = bool(options.get("case_sensitive", False))
        self._preserve_formatting = bool(options.get("preserve_formatting", False))
        self._prompt_on_ambiguous = bool(options.get("prompt_on_ambiguous", True))

        self._canonical_lookup: dict[str, str] = {}
        self._alias_lookup: dict[str, str] = {}
        self._misspelling_lookup: dict[str, str] = {}
        self._canonical_metadata: dict[str, dict[str, Any]] = {}
        self._canonical_normalised: dict[str, str] = {}
        self._fuzzy_candidates: list[tuple[str, str]] = []

        self._build_lookup_tables()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def canonical_names(self) -> list[str]:
        return sorted(self._canonical_metadata.keys(), key=str.casefold)

    @property
    def auto_correct_threshold(self) -> float:
        return self._auto_threshold

    @property
    def fuzzy_match_cutoff(self) -> float:
        return self._fuzzy_cutoff

    def should_auto_correct(self, result: NameCorrectionResult) -> bool:
        """Return True when automatic correction should be applied."""
        return (
            result.canonical is not None
            and result.changed
            and result.confidence >= self._auto_threshold
        )

    def record_correction(self, result: NameCorrectionResult) -> None:
        """Persist and log a correction that has been applied elsewhere."""
        self._log_correction(result)

    def correct_name(
        self,
        name: str,
        *,
        log_change: bool = False,
    ) -> NameCorrectionResult:
        """Return the canonical representation of ``name``."""
        original_input = name or ""
        cleaned_input = original_input.strip()

        if not cleaned_input:
            return NameCorrectionResult(
                original_input,
                "",
                "",
                None,
                "empty",
                0.0,
                {},
            )

        if not self._enabled:
            return NameCorrectionResult(
                original_input,
                cleaned_input,
                cleaned_input,
                cleaned_input,
                "disabled",
                1.0,
                {},
            )

        normalised = self._normalise(cleaned_input)
        canonical, method, confidence = self._lookup(normalised)

        if canonical is None:
            return NameCorrectionResult(
                original_input,
                cleaned_input,
                cleaned_input,
                None,
                method,
                confidence,
                {},
            )

        corrected = (
            canonical
            if not self._preserve_formatting
            else self._match_format(cleaned_input, canonical)
        )
        metadata = dict(self._canonical_metadata.get(canonical, {}))

        result = NameCorrectionResult(
            original_input,
            cleaned_input,
            corrected,
            canonical,
            method,
            confidence,
            metadata,
        )

        if log_change and result.changed:
            self._log_correction(result)

        return result

    def correct_transcript(
        self,
        transcript: str,
        *,
        return_details: bool = False,
    ) -> str | tuple[str, list[NameCorrectionResult]]:
        """Auto-correct speaker labels in a transcript."""
        if not transcript:
            return ("", []) if return_details else ""

        applied: list[NameCorrectionResult] = []

        def _replacer(match: re.Match[str]) -> str:
            prefix = match.group("prefix") or ""
            raw_name = match.group("name")
            separator = match.group("separator")

            result = self.correct_name(raw_name, log_change=False)
            if not self.should_auto_correct(result):
                return match.group(0)

            applied.append(result)
            self._log_correction(result)
            return f"{prefix}{result.corrected}{separator}"

        corrected_text, _ = SPEAKER_PATTERN.subn(_replacer, transcript)

        if return_details:
            return corrected_text, applied
        return corrected_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_lookup_tables(self) -> None:
        """Populate lookup dictionaries from show config and voice bank."""
        self._ingest_show_config()
        self._ingest_voice_bank()
        self._fuzzy_candidates = [
            (canonical, normalised) for canonical, normalised in self._canonical_normalised.items()
        ]

    def _ingest_show_config(self) -> None:
        """Load canonical names, aliases, and misspellings from show config."""
        for entry in self.show_config.get("cast_members", []) or []:
            canonical = entry.get("canonical_name")
            if not canonical:
                continue

            normalised = self._normalise(canonical)
            if not normalised:
                continue

            self._register_canonical(
                canonical,
                normalised,
                {
                    "role": entry.get("role", "other"),
                    "source": "show_config",
                },
            )

            aliases = entry.get("aliases") or []
            for alias in aliases:
                self._register_alias(alias, canonical)

            misspellings = entry.get("common_misspellings") or []
            for typo in misspellings:
                self._register_misspelling(typo, canonical)

    def _ingest_voice_bank(self) -> None:
        """Load supplemental metadata from the voice bank."""
        if self.voice_bank is None:
            return

        for profile in self.voice_bank.list_speakers():
            canonical = profile.display_name
            if not canonical:
                continue

            normalised = self._normalise(canonical)
            if not normalised:
                continue

            self._register_canonical(
                canonical,
                normalised,
                {
                    "source": "voice_bank",
                    "speaker_key": profile.key,
                },
            )

            for alias in profile.common_aliases:
                self._register_alias(alias, canonical)

            for typo in profile.common_misspellings:
                self._register_misspelling(typo, canonical)

    def _register_canonical(
        self,
        canonical: str,
        normalised: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Add canonical mappings and metadata."""
        if normalised not in self._canonical_lookup:
            self._canonical_lookup[normalised] = canonical
        self._canonical_normalised[canonical] = normalised
        merged = self._canonical_metadata.setdefault(canonical, {})
        merged.update({k: v for k, v in metadata.items() if v is not None})

    def _register_alias(self, alias: str, canonical: str) -> None:
        """Add alias lookup entry."""
        normalised = self._normalise(alias)
        if normalised and normalised not in self._alias_lookup:
            self._alias_lookup[normalised] = canonical

    def _register_misspelling(self, misspelling: str, canonical: str) -> None:
        """Add misspelling lookup entry."""
        normalised = self._normalise(misspelling)
        if normalised and normalised not in self._misspelling_lookup:
            self._misspelling_lookup[normalised] = canonical

    def _normalise(self, value: str) -> str:
        """Return a normalised representation used for lookups."""
        text = unicodedata.normalize("NFKD", value or "").strip()
        if not self._case_sensitive:
            text = text.casefold()
        # Keep alphanumeric characters only, removing spaces/punctuation.
        return "".join(ch for ch in text if ch.isalnum())

    def _lookup(self, normalised: str) -> tuple[str | None, str, float]:
        """Return the canonical name using hierarchical lookup strategies."""
        if not normalised:
            return None, "invalid", 0.0

        canonical = self._canonical_lookup.get(normalised)
        if canonical:
            return canonical, "canonical", 1.0

        canonical = self._alias_lookup.get(normalised)
        if canonical:
            return canonical, "alias", 0.95

        canonical = self._misspelling_lookup.get(normalised)
        if canonical:
            return canonical, "misspelling", 0.9

        return self._fuzzy_match(normalised)

    def _fuzzy_match(self, normalised: str) -> tuple[str | None, str, float]:
        """Attempt fuzzy matching by string similarity."""
        best_name: str | None = None
        best_score = 0.0

        for canonical, canonical_norm in self._fuzzy_candidates:
            score = SequenceMatcher(None, normalised, canonical_norm).ratio()
            if score > best_score:
                best_name = canonical
                best_score = score

        if best_name and best_score >= self._fuzzy_cutoff:
            return best_name, "fuzzy", best_score

        return None, "no_match", best_score

    def _match_format(self, original: str, canonical: str) -> str:
        """Attempt to preserve the case format of the original string."""
        if not original:
            return canonical
        if original.isupper():
            return canonical.upper()
        if original.islower():
            return canonical.lower()
        if original.istitle():
            return canonical.title()
        return canonical

    def _log_correction(self, result: NameCorrectionResult) -> None:
        """Emit logging and persist the correction when configured."""
        if not self._log_enabled or not result.changed:
            return

        LOGGER.info(
            "Name correction applied: '%s' -> '%s' via %s (confidence=%.2f)",
            result.cleaned_input,
            result.corrected,
            result.method,
            result.confidence,
        )

        if self.voice_bank is not None:
            metadata = {
                "canonical": result.canonical,
                "method": result.method,
                **result.metadata,
            }
            self.voice_bank.log_name_correction(
                original=result.cleaned_input,
                corrected=result.corrected,
                method=result.method,
                confidence=result.confidence,
                metadata=metadata,
            )
