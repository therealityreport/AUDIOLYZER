"""Speaker identification pipeline backed by the persisted voice bank."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from ...storage.db import SQLiteDatabase
from ...storage.paths import PathsConfig
from ...storage.voice_bank_manager import SpeakerProfile, VoiceBankManager
from ...utils.audio_io import AudioClip, extract_segment, load_audio, resample_audio
from ...utils.logging import get_logger
from ..alignment.align_asr_diar import AlignmentResult
from ..diarization.pyannote_pipeline import DiarizationResult, DiarizationSegment
from ..embeddings.pyannote_embeddings import (
    PyannoteEmbeddingError,
    build_pyannote_embedding_backend,
)

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from resemblyzer import VoiceEncoder
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    VoiceEncoder = None

__all__ = [
    "SpeakerAssignment",
    "SpeakerIdentificationError",
    "SpeakerIdentificationResult",
    "VoiceBankPipeline",
    "build_voice_bank_pipeline",
]


class SpeakerIdentificationError(RuntimeError):
    """Raised when speaker identification cannot be executed."""


@dataclass(slots=True)
class SpeakerAssignment:
    """Represents the outcome of matching a diarization cluster."""

    cluster_id: str
    duration_seconds: float
    segment_count: int
    similarity: float | None = None
    speaker_key: str | None = None
    display_name: str | None = None
    embedding_path: Path | None = None
    auto_registered: bool = False
    confidence_level: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.speaker_key is not None

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""
        payload: dict[str, object] = {
            "cluster_id": self.cluster_id,
            "duration_seconds": self.duration_seconds,
            "segment_count": self.segment_count,
            "matched": self.matched,
            "similarity": self.similarity,
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "notes": list(self.notes),
        }
        if self.embedding_path is not None:
            payload["embedding_path"] = str(self.embedding_path)
        payload["auto_registered"] = self.auto_registered
        if self.confidence_level is not None:
            payload["confidence_level"] = self.confidence_level
        return payload


@dataclass(slots=True)
class SpeakerIdentificationResult:
    """Result returned by the voice bank pipeline."""

    episode_id: str
    assignments: list[SpeakerAssignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }

    @property
    def matched(self) -> int:
        return sum(1 for assignment in self.assignments if assignment.matched)

    @property
    def unmatched(self) -> int:
        return sum(1 for assignment in self.assignments if not assignment.matched)


@dataclass(slots=True)
class _VoiceIdentificationSettings:
    """Configuration extracted from project settings."""

    embedding_provider: str
    high_confidence_threshold: float
    medium_confidence_threshold: float
    low_confidence_threshold: float
    auto_register_new_speakers: bool
    min_samples_required: int
    min_sample_duration_seconds: float
    max_embeddings_per_speaker: int

    def classify_similarity(self, value: float) -> str | None:
        """Return the confidence bucket for ``value`` or None if below threshold."""
        if value >= self.high_confidence_threshold:
            return "high"
        if value >= self.medium_confidence_threshold:
            return "medium"
        if value >= self.low_confidence_threshold:
            return "low"
        return None


class _EmbeddingBackend(Protocol):
    """Protocol describing embedding backends used by the voice bank pipeline."""

    target_sample_rate: int

    def encode(self, clip: AudioClip) -> np.ndarray: ...


class _ResemblyzerBackend:
    """Thin wrapper around the Resemblyzer encoder."""

    def __init__(self, *, target_sample_rate: int = 16000) -> None:
        if VoiceEncoder is None:
            raise SpeakerIdentificationError(
                "Resemblyzer is not installed; install the 'resemblyzer' extra to enable "
                "speaker identification."
            )
        self.target_sample_rate = target_sample_rate
        self._encoder = VoiceEncoder()

    def encode(self, clip: AudioClip) -> np.ndarray:
        """Return an embedding vector for the given clip."""
        waveform = clip.samples
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0, dtype=np.float32)

        sample_rate = clip.sample_rate
        if sample_rate != self.target_sample_rate:
            waveform = resample_audio(
                waveform,
                sample_rate,
                self.target_sample_rate,
                method="sinc_best",
            )
            sample_rate = self.target_sample_rate

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        embedding = self._encoder.embed_utterance(waveform)
        return embedding.astype(np.float32, copy=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity with defensive checks."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VoiceBankPipeline:
    """Matches diarization clusters to known speakers using the voice bank."""

    def __init__(
        self,
        config: Mapping[str, object],
        paths: PathsConfig,
        *,
        embedding_backend: _EmbeddingBackend | None = None,
        voice_bank: VoiceBankManager | None = None,
    ) -> None:
        self.paths = paths
        self._settings = self._load_settings(config)
        self._embedding_backend = embedding_backend or self._build_embedding_backend(config)
        self._voice_bank_manager = voice_bank or self._build_voice_bank_manager()
        self._embedding_root = paths.data_root / "voice_bank" / "embeddings"
        self._embedding_root.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_backend(self) -> _EmbeddingBackend:
        """Return the embedding backend used for speaker comparisons."""
        return self._embedding_backend

    @property
    def settings(self) -> _VoiceIdentificationSettings:
        """Return the resolved identification settings."""
        return self._settings

    @property
    def embedding_root(self) -> Path:
        """Directory where per-speaker embeddings are stored."""
        return self._embedding_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def identify(
        self,
        *,
        episode_id: str,
        audio_path: Path,
        diarization: DiarizationResult,
        alignment: AlignmentResult,
    ) -> SpeakerIdentificationResult:
        """Return speaker assignments for each diarization cluster."""
        clip = load_audio(
            audio_path,
            target_sample_rate=self._embedding_backend.target_sample_rate,
        )
        cluster_segments = self._group_segments_by_speaker(diarization)

        # Filter clusters that do not meet minimum requirements.
        valid_clusters = {
            cluster_id: segments
            for cluster_id, segments in cluster_segments.items()
            if self._cluster_is_valid(segments)
        }

        existing_embeddings = self._load_voice_bank_embeddings()

        assignments: list[SpeakerAssignment] = []
        for cluster_id, segments in sorted(valid_clusters.items()):
            waveform = self._collect_waveform(clip, segments)
            assignment = SpeakerAssignment(
                cluster_id=cluster_id,
                duration_seconds=sum(max(seg.end - seg.start, 0.0) for seg in segments),
                segment_count=len(segments),
            )

            if waveform is None or waveform.size == 0:
                assignment.notes.append("No usable audio for cluster.")
                assignments.append(assignment)
                continue

            embedding_clip = AudioClip(samples=waveform, sample_rate=clip.sample_rate)
            embedding_vector = self._embedding_backend.encode(embedding_clip)

            match = self._match_embedding(embedding_vector, existing_embeddings)
            if match is None:
                if self._settings.auto_register_new_speakers:
                    profile = self._register_new_speaker(cluster_id)
                    assignment.speaker_key = profile.key
                    assignment.display_name = profile.display_name
                    assignment.auto_registered = True
                    embedding_path = self._persist_embedding(
                        profile=profile,
                        episode_id=episode_id,
                        cluster_id=cluster_id,
                        embedding_vector=embedding_vector,
                        alignment=alignment,
                    )
                    assignment.embedding_path = embedding_path
                    if profile.id is not None:
                        existing_embeddings.setdefault(profile.id, []).append(embedding_vector)
                else:
                    assignment.notes.append(
                        "No speaker met the minimum similarity threshold; manual labeling required."
                    )
                assignments.append(assignment)
                continue

            profile, similarity, confidence_level = match
            assignment.speaker_key = profile.key
            assignment.display_name = profile.display_name
            assignment.similarity = similarity
            assignment.confidence_level = confidence_level
            if confidence_level in {"medium", "low"}:
                assignment.notes.append(
                    f"Similarity {similarity:.3f} classified as {confidence_level}; review recommended."
                )

            embedding_path = self._persist_embedding(
                profile=profile,
                episode_id=episode_id,
                cluster_id=cluster_id,
                embedding_vector=embedding_vector,
                alignment=alignment,
            )
            assignment.embedding_path = embedding_path

            self._voice_bank_manager.increment_segment_count(profile.key, increment=len(segments))
            assignments.append(assignment)

        result = SpeakerIdentificationResult(episode_id=episode_id, assignments=assignments)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_voice_bank_manager(self) -> VoiceBankManager:
        database = SQLiteDatabase(self.paths.voice_bank_db)
        if not database.db_path.exists():
            LOGGER.info("Voice bank database not found; initialising at %s", database.db_path)
            database.initialize()
        return VoiceBankManager(database)

    def _build_embedding_backend(self, config: Mapping[str, object]) -> _EmbeddingBackend:
        provider = self._settings.embedding_provider
        if provider == "pyannote":
            try:
                return build_pyannote_embedding_backend(config)
            except PyannoteEmbeddingError as exc:
                raise SpeakerIdentificationError(
                    "Pyannote embeddings are not available. Ensure pyannote.audio is installed "
                    "and the authentication token is configured. Details: "
                    f"{exc}"
                ) from exc
        if provider == "resemblyzer":
            return _ResemblyzerBackend()
        raise SpeakerIdentificationError(f"Unsupported embedding provider '{provider}'.")

    @staticmethod
    def _group_segments_by_speaker(
        diarization: DiarizationResult,
    ) -> dict[str, list[DiarizationSegment]]:
        grouped: dict[str, list[DiarizationSegment]] = defaultdict(list)
        for segment in diarization.segments:
            grouped[str(segment.speaker)].append(segment)
        return grouped

    def _cluster_is_valid(self, segments: Sequence[DiarizationSegment]) -> bool:
        if len(segments) < self._settings.min_samples_required:
            return False
        duration = sum(max(segment.end - segment.start, 0.0) for segment in segments)
        return duration >= self._settings.min_sample_duration_seconds

    def _collect_waveform(
        self,
        clip: AudioClip,
        segments: Sequence[DiarizationSegment],
    ) -> np.ndarray | None:
        chunks: list[np.ndarray] = []
        for segment in segments:
            try:
                extract = extract_segment(clip, segment.start, segment.end)
            except ValueError as exc:  # pragma: no cover - defensive
                LOGGER.debug("Failed to extract segment %s: %s", segment, exc)
                continue
            if extract.samples.size == 0:
                continue
            waveform = extract.samples
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0, dtype=np.float32)
            chunks.append(waveform.astype(np.float32, copy=False))
        if not chunks:
            return None
        return np.concatenate(chunks, axis=-1)

    def _load_voice_bank_embeddings(self) -> dict[int, list[np.ndarray]]:
        embeddings: dict[int, list[np.ndarray]] = {}
        for profile in self._voice_bank_manager.list_speakers():
            if profile.id is None:
                continue
            vectors: list[np.ndarray] = []
            for entry in self._voice_bank_manager.list_embeddings(profile.id):
                path = entry.embedding_path
                if not path.exists():
                    LOGGER.debug("Skipping missing embedding file %s", path)
                    continue
                try:
                    vector = np.load(path, allow_pickle=False)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load embedding %s: %s", path, exc)
                    continue
                if vector.ndim != 1:
                    LOGGER.debug("Skipping malformed embedding %s", path)
                    continue
                vectors.append(vector.astype(np.float32, copy=False))
            if vectors:
                embeddings[profile.id] = vectors
        return embeddings

    def _match_embedding(
        self,
        embedding: np.ndarray,
        candidates: Mapping[int, list[np.ndarray]],
    ) -> tuple[SpeakerProfile, float, str] | None:
        best_profile: SpeakerProfile | None = None
        best_similarity = -math.inf

        for profile in self._voice_bank_manager.list_speakers():
            if profile.id is None or profile.id not in candidates:
                continue
            for candidate in candidates[profile.id]:
                similarity = _cosine_similarity(embedding, candidate)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_profile = profile

        if best_profile is None:
            return None
        confidence = self._settings.classify_similarity(best_similarity)
        if confidence is None:
            return None
        return best_profile, best_similarity, confidence

    def _register_new_speaker(self, cluster_id: str) -> SpeakerProfile:
        display_name = f"Unknown Speaker {cluster_id}"
        key = f"auto_{cluster_id}".lower()
        profile = self._voice_bank_manager.upsert_speaker(
            display_name=display_name,
            key=key,
            aliases=[cluster_id],
            notes="Auto-registered by speaker identification pipeline.",
        )
        LOGGER.info(
            "Auto-registered speaker '%s' (key=%s) for cluster %s.",
            profile.display_name,
            profile.key,
            cluster_id,
        )
        return profile

    def _persist_embedding(
        self,
        *,
        profile: SpeakerProfile,
        episode_id: str,
        cluster_id: str,
        embedding_vector: np.ndarray,
        alignment: AlignmentResult,
    ) -> Path:
        if profile.id is None:
            raise SpeakerIdentificationError("Cannot persist embedding without speaker id.")

        speaker_dir = self._embedding_root / profile.key
        speaker_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = speaker_dir / f"{episode_id}_{cluster_id}.npy"
        np.save(embedding_path, embedding_vector.astype(np.float32, copy=False))

        metadata = alignment.metadata.to_dict()
        LOGGER.debug(
            "Persisted embedding for %s at %s (metadata=%s)",
            profile.display_name,
            embedding_path,
            metadata,
        )

        self._voice_bank_manager.add_embedding(
            speaker_id=profile.id,
            embedding_path=embedding_path,
            source_episode=episode_id,
        )
        self._enforce_embedding_capacity(profile.id)
        return embedding_path

    def _enforce_embedding_capacity(self, speaker_id: int) -> None:
        entries = self._voice_bank_manager.list_embeddings(speaker_id)
        if len(entries) <= self._settings.max_embeddings_per_speaker:
            return

        # Remove oldest embeddings beyond capacity.
        for entry in entries[self._settings.max_embeddings_per_speaker :]:
            if entry.id is not None:
                self._voice_bank_manager.remove_embedding(entry.id)
            try:
                entry.embedding_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                logging.getLogger(__name__).debug(
                    "Unable to delete embedding file %s", entry.embedding_path
                )

    @staticmethod
    def _load_settings(config: Mapping[str, object]) -> _VoiceIdentificationSettings:
        voice_ident_cfg = config.get("voice_identification") or {}
        if not isinstance(voice_ident_cfg, Mapping):
            voice_ident_cfg = {}

        voice_bank_cfg = config.get("voice_bank") or {}
        if not isinstance(voice_bank_cfg, Mapping):
            voice_bank_cfg = {}

        provider = str(voice_ident_cfg.get("embedding_provider", "pyannote")).strip().lower()
        if provider not in {"pyannote", "resemblyzer"}:
            provider = "pyannote"

        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        def _coerce(value: object, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        similarity_threshold = voice_ident_cfg.get("similarity_threshold")
        high_default = (
            0.85 if similarity_threshold is None else _clamp(_coerce(similarity_threshold, 0.85))
        )
        high = _clamp(_coerce(voice_ident_cfg.get("high_confidence_threshold"), high_default))
        medium_default = min(high, 0.75)
        medium = _clamp(_coerce(voice_ident_cfg.get("medium_confidence_threshold"), medium_default))
        low_default = min(medium, 0.6)
        low = _clamp(_coerce(voice_ident_cfg.get("low_confidence_threshold"), low_default))

        if medium > high:
            medium = high
        if low > medium:
            low = medium

        auto_register = bool(voice_ident_cfg.get("auto_register_new_speakers", False))
        min_samples_required = int(voice_ident_cfg.get("min_samples_required", 3))
        min_duration = float(voice_bank_cfg.get("min_sample_duration_seconds", 1.0))
        max_embeddings = int(voice_bank_cfg.get("max_embeddings_per_speaker", 20))

        return _VoiceIdentificationSettings(
            embedding_provider=provider,
            high_confidence_threshold=high,
            medium_confidence_threshold=medium,
            low_confidence_threshold=low,
            auto_register_new_speakers=auto_register,
            min_samples_required=min_samples_required,
            min_sample_duration_seconds=min_duration,
            max_embeddings_per_speaker=max_embeddings,
        )


def build_voice_bank_pipeline(
    config: Mapping[str, object],
    paths: PathsConfig,
) -> VoiceBankPipeline:
    """Factory that constructs a voice bank pipeline instance."""
    return VoiceBankPipeline(config, paths)
