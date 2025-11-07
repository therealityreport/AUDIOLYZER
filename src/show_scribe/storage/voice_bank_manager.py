"""Utilities for reading and updating the voice bank database."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datetime import datetime
from zoneinfo import ZoneInfo

from .db import DatabaseError, SQLiteDatabase

__all__ = [
    "SpeakerProfile",
    "VoiceBankManager",
    "VoiceEmbedding",
]


def _to_json_text(values: Iterable[str]) -> str:
    """Serialise an iterable of strings into canonical JSON list text."""
    return json.dumps(sorted({value.strip() for value in values if value.strip()}))


def _from_json_text(raw: str | bytes | None) -> tuple[str, ...]:
    """Deserialize JSON stored in TEXT columns into an immutable tuple."""
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if isinstance(data, list):
        return tuple(str(item) for item in data if isinstance(item, str))
    return ()


def normalize_key(name: str) -> str:
    """Normalise a display name into the lowercase key stored in the database."""
    return "".join(character for character in name.lower() if character.isalnum())


def _row_value(row: Any, key: str) -> Any:
    if hasattr(row, "keys"):
        keys = row.keys()
        if key in keys:
            return row[key]
    try:
        return row[key]
    except Exception:
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=ZoneInfo("UTC"))
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


@dataclass(slots=True)
class SpeakerProfile:
    """Canonical view of a speaker profile row."""

    id: int | None
    key: str
    display_name: str
    common_aliases: tuple[str, ...] = field(default_factory=tuple)
    common_misspellings: tuple[str, ...] = field(default_factory=tuple)
    phonetic_spelling: str | None = None
    first_appearance_episode: str | None = None
    total_segments: int = 0
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class VoiceEmbedding:
    """Metadata that describes a stored voice embedding file."""

    id: int | None
    speaker_id: int
    embedding_path: Path
    source_episode: str | None
    source_timestamp: str | None
    audio_sample_path: Path | None
    quality_score: float | None
    created_at: datetime | None


class VoiceBankManager:
    """Facade that encapsulates common voice bank operations."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # Speaker profile operations
    # ------------------------------------------------------------------
    def list_speakers(self) -> list[SpeakerProfile]:
        """Return all speaker profiles sorted by display name."""
        rows = self._db.fetch_all(
            """
            SELECT
                id,
                name,
                display_name,
                common_aliases,
                common_misspellings,
                phonetic_spelling,
                first_appearance_episode,
                total_segments,
                notes,
                created_date,
                last_updated
            FROM speaker_profiles
            ORDER BY display_name COLLATE NOCASE;
            """
        )
        return [self._row_to_profile(row) for row in rows]

    def get_speaker_by_key(self, key: str) -> SpeakerProfile | None:
        """Return a speaker profile by its canonical key."""
        row = self._db.fetch_one(
            """
            SELECT
                id,
                name,
                display_name,
                common_aliases,
                common_misspellings,
                phonetic_spelling,
                first_appearance_episode,
                total_segments,
                notes,
                created_date,
                last_updated
            FROM speaker_profiles
            WHERE name = ?;
            """,
            (key,),
        )
        return self._row_to_profile(row) if row else None

    def get_speaker_by_display_name(self, display_name: str) -> SpeakerProfile | None:
        """Return a speaker profile matched by display name."""
        row = self._db.fetch_one(
            """
            SELECT
                id,
                name,
                display_name,
                common_aliases,
                common_misspellings,
                phonetic_spelling,
                first_appearance_episode,
                total_segments,
                notes,
                created_date,
                last_updated
            FROM speaker_profiles
            WHERE display_name = ? COLLATE NOCASE;
            """,
            (display_name,),
        )
        return self._row_to_profile(row) if row else None

    def upsert_speaker(
        self,
        *,
        display_name: str,
        key: str | None = None,
        aliases: Sequence[str] | None = None,
        misspellings: Sequence[str] | None = None,
        phonetic_spelling: str | None = None,
        first_appearance_episode: str | None = None,
        notes: str | None = None,
    ) -> SpeakerProfile:
        """Insert or update a speaker profile."""
        normalized_key = key or normalize_key(display_name)
        alias_text = _to_json_text(aliases or ())
        misspelling_text = _to_json_text(misspellings or ())

        with self._db.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM speaker_profiles WHERE name = ?;",
                (normalized_key,),
            ).fetchone()

            if existing:
                connection.execute(
                    """
                    UPDATE speaker_profiles
                    SET display_name = ?,
                        common_aliases = ?,
                        common_misspellings = ?,
                        phonetic_spelling = ?,
                        first_appearance_episode = ?,
                        notes = ?
                    WHERE id = ?;
                    """,
                    (
                        display_name,
                        alias_text,
                        misspelling_text,
                        phonetic_spelling,
                        first_appearance_episode,
                        notes,
                        existing["id"],
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO speaker_profiles (
                        name,
                        display_name,
                        common_aliases,
                        common_misspellings,
                        phonetic_spelling,
                        first_appearance_episode,
                        notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        normalized_key,
                        display_name,
                        alias_text,
                        misspelling_text,
                        phonetic_spelling,
                        first_appearance_episode,
                        notes,
                    ),
                )
        profile = self.get_speaker_by_key(normalized_key)
        if profile is None:  # pragma: no cover - defensive
            raise DatabaseError("Failed to persist speaker profile.")
        return profile

    def increment_segment_count(self, key: str, *, increment: int = 1) -> None:
        """Increase the segment counter used for analytics."""
        with self._db.connect() as connection:
            connection.execute(
                """
                UPDATE speaker_profiles
                SET total_segments = total_segments + ?
                WHERE name = ?;
                """,
                (increment, key),
            )

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------
    def add_embedding(
        self,
        *,
        speaker_id: int,
        embedding_path: Path,
        source_episode: str | None = None,
        source_timestamp: str | None = None,
        audio_sample_path: Path | None = None,
        quality_score: float | None = None,
    ) -> VoiceEmbedding:
        """Register a new voice embedding entry."""
        with self._db.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO voice_embeddings (
                    speaker_id,
                    embedding_path,
                    source_episode,
                    source_timestamp,
                    audio_sample_path,
                    quality_score
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    speaker_id,
                    str(embedding_path),
                    source_episode,
                    source_timestamp,
                    str(audio_sample_path) if audio_sample_path else None,
                    quality_score,
                ),
            )
            embedding_id = cursor.lastrowid

        return VoiceEmbedding(
            id=embedding_id,
            speaker_id=speaker_id,
            embedding_path=embedding_path,
            source_episode=source_episode,
            source_timestamp=source_timestamp,
            audio_sample_path=audio_sample_path,
            quality_score=quality_score,
            created_at=datetime.now(ZoneInfo("UTC")),
        )

    def list_embeddings(self, speaker_id: int) -> list[VoiceEmbedding]:
        """Return metadata for embeddings associated with a speaker."""
        rows = self._db.fetch_all(
            """
            SELECT
                id,
                speaker_id,
                embedding_path,
                source_episode,
                source_timestamp,
                audio_sample_path,
                quality_score,
                created_date
            FROM voice_embeddings
            WHERE speaker_id = ?
            ORDER BY created_date DESC;
            """,
            (speaker_id,),
        )
        return [
            VoiceEmbedding(
                id=row["id"],
                speaker_id=row["speaker_id"],
                embedding_path=Path(row["embedding_path"]),
                source_episode=row["source_episode"],
                source_timestamp=row["source_timestamp"],
                audio_sample_path=(
                    Path(row["audio_sample_path"]) if row["audio_sample_path"] else None
                ),
                quality_score=row["quality_score"],
                created_at=_parse_timestamp(_row_value(row, "created_date")),
            )
            for row in rows
        ]

    def remove_embedding(self, embedding_id: int) -> None:
        """Delete an embedding entry."""
        self._db.execute(
            "DELETE FROM voice_embeddings WHERE id = ?;",
            (embedding_id,),
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def log_name_correction(
        self,
        *,
        original: str,
        corrected: str,
        method: str,
        confidence: float | None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Persist a record of automated name corrections."""
        metadata_text = json.dumps(metadata or {}, sort_keys=True)
        with self._db.connect() as connection:
            connection.execute(
                """
                INSERT INTO name_corrections_log (
                    original_text,
                    corrected_text,
                    method,
                    confidence,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (original, corrected, method, confidence, metadata_text),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_profile(row: Any) -> SpeakerProfile:
        """Convert a database row into a SpeakerProfile dataclass."""
        return SpeakerProfile(
            id=row["id"],
            key=row["name"],
            display_name=row["display_name"],
            common_aliases=_from_json_text(row["common_aliases"]),
            common_misspellings=_from_json_text(row["common_misspellings"]),
            phonetic_spelling=row["phonetic_spelling"],
            first_appearance_episode=row["first_appearance_episode"],
            total_segments=row["total_segments"] or 0,
            notes=row["notes"],
            created_at=_parse_timestamp(_row_value(row, "created_date")),
            updated_at=_parse_timestamp(_row_value(row, "last_updated")),
        )
