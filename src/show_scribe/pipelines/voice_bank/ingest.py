"""Voice bank ingestion service for external audio and manual assignments."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...storage.voice_bank_manager import SpeakerProfile, VoiceBankManager, normalize_key
from ...utils.audio_io import AudioClip, extract_segment, load_audio, save_audio
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "IngestionResult",
    "IngestionSummary",
    "ingest_segment_embedding",
    "compute_audio_quality_score",
]


@dataclass(slots=True)
class IngestionResult:
    """Result of ingesting a single audio segment."""

    speaker_display_name: str
    start: float
    end: float
    duration: float
    accepted: bool
    rejection_reason: str | None = None
    embedding_id: int | None = None
    embedding_path: Path | None = None
    audio_sample_path: Path | None = None
    quality_score: float | None = None


@dataclass(slots=True)
class IngestionSummary:
    """Summary of ingestion operation across all speakers."""

    results: list[IngestionResult] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return sum(1 for r in self.results if r.accepted)

    @property
    def rejected(self) -> int:
        return sum(1 for r in self.results if not r.accepted)

    def by_speaker(self) -> dict[str, list[IngestionResult]]:
        """Group results by speaker."""
        grouped: dict[str, list[IngestionResult]] = {}
        for result in self.results:
            grouped.setdefault(result.speaker_display_name, []).append(result)
        return grouped


def compute_audio_quality_score(clip: AudioClip) -> float:
    """Compute a simple quality score for an audio clip (0.0 to 1.0).

    Considers:
    - RMS energy (too quiet = low quality)
    - Peak clipping detection
    - Dynamic range
    """
    waveform = clip.samples
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    if waveform.size == 0:
        return 0.0

    # RMS energy
    rms = float(np.sqrt(np.mean(waveform**2)))
    rms_score = min(rms / 0.1, 1.0)  # Normalize around 0.1 RMS as good

    # Clipping detection (samples at or near ±1.0)
    clipping_threshold = 0.99
    clipped = np.sum(np.abs(waveform) >= clipping_threshold)
    clipping_ratio = clipped / waveform.size
    clipping_score = max(0.0, 1.0 - clipping_ratio * 10)  # Heavy penalty for clipping

    # Dynamic range (std dev as proxy)
    std_dev = float(np.std(waveform))
    dynamic_score = min(std_dev / 0.1, 1.0)

    # Weighted average
    quality = (rms_score * 0.4) + (clipping_score * 0.4) + (dynamic_score * 0.2)
    return max(0.0, min(1.0, quality))


def trim_silence(
    clip: AudioClip,
    *,
    threshold_db: float = -40.0,
    min_duration_seconds: float = 0.1,
) -> AudioClip:
    """Trim leading and trailing silence from an audio clip.

    Simple energy-based trimming without full VAD.
    """
    waveform = clip.samples
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    if waveform.size == 0:
        return clip

    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20.0)

    # Find first and last sample above threshold
    above_threshold = np.abs(waveform) > threshold_linear
    if not np.any(above_threshold):
        # Entire clip is silence
        return AudioClip(samples=np.array([], dtype=np.float32), sample_rate=clip.sample_rate)

    first_idx = int(np.argmax(above_threshold))
    last_idx = len(waveform) - int(np.argmax(above_threshold[::-1])) - 1

    # Ensure minimum duration
    min_samples = int(min_duration_seconds * clip.sample_rate)
    if last_idx - first_idx < min_samples:
        return AudioClip(samples=np.array([], dtype=np.float32), sample_rate=clip.sample_rate)

    trimmed = waveform[first_idx : last_idx + 1]
    return AudioClip(samples=trimmed, sample_rate=clip.sample_rate)


def ingest_segment_embedding(
    *,
    audio_path: Path,
    speaker_profile: SpeakerProfile,
    start: float,
    end: float,
    voice_bank_manager: VoiceBankManager,
    embedding_backend: Any,  # _EmbeddingBackend protocol
    voice_bank_root: Path,
    config: Mapping[str, Any],
    source_episode: str | None = None,
) -> IngestionResult:
    """Ingest a single audio segment into the voice bank.

    Args:
        audio_path: Path to source audio file (should be 16kHz mono enhanced vocals)
        speaker_profile: Target speaker profile
        start: Start time in seconds
        end: End time in seconds
        voice_bank_manager: Voice bank manager instance
        embedding_backend: Embedding encoder (must have .encode() method)
        voice_bank_root: Root directory for voice bank storage
        config: Configuration dict
        source_episode: Optional episode ID for tracking

    Returns:
        IngestionResult with acceptance status and details
    """
    duration = end - start

    # Extract config parameters
    voice_bank_cfg = config.get("voice_bank", {})
    min_duration = float(voice_bank_cfg.get("min_sample_duration_seconds", 1.0))
    max_embeddings = int(voice_bank_cfg.get("max_embeddings_per_speaker", 20))

    # Validate duration
    if duration < min_duration:
        return IngestionResult(
            speaker_display_name=speaker_profile.display_name,
            start=start,
            end=end,
            duration=duration,
            accepted=False,
            rejection_reason=f"Duration {duration:.2f}s < minimum {min_duration}s",
        )

    try:
        # Load and extract segment
        clip = load_audio(audio_path, target_sample_rate=16000, mono=True)
        segment = extract_segment(clip, start, end)

        # Trim silence
        trimmed = trim_silence(segment, threshold_db=-40.0, min_duration_seconds=min_duration)

        if trimmed.samples.size == 0 or trimmed.duration_seconds < min_duration:
            return IngestionResult(
                speaker_display_name=speaker_profile.display_name,
                start=start,
                end=end,
                duration=duration,
                accepted=False,
                rejection_reason="Segment is entirely silence after trimming",
            )

        # Compute quality score
        quality_score = compute_audio_quality_score(trimmed)

        if quality_score < 0.2:
            return IngestionResult(
                speaker_display_name=speaker_profile.display_name,
                start=start,
                end=end,
                duration=duration,
                accepted=False,
                rejection_reason=f"Quality score {quality_score:.2f} too low",
                quality_score=quality_score,
            )

        # Generate embedding
        try:
            embedding_vector = embedding_backend.encode(trimmed)
        except Exception as exc:
            LOGGER.warning(
                "Failed to generate embedding for %s: %s", speaker_profile.display_name, exc
            )
            return IngestionResult(
                speaker_display_name=speaker_profile.display_name,
                start=start,
                end=end,
                duration=duration,
                accepted=False,
                rejection_reason=f"Embedding generation failed: {exc}",
                quality_score=quality_score,
            )

        if speaker_profile.id is None:
            return IngestionResult(
                speaker_display_name=speaker_profile.display_name,
                start=start,
                end=end,
                duration=duration,
                accepted=False,
                rejection_reason="Speaker profile has no ID",
                quality_score=quality_score,
            )

        # Save embedding
        embeddings_dir = voice_bank_root / "embeddings" / speaker_profile.key
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        embedding_id_str = str(uuid.uuid4())[:8]
        embedding_path = embeddings_dir / f"{source_episode}_{embedding_id_str}.npy"
        np.save(embedding_path, embedding_vector.astype(np.float32, copy=False))

        # Optionally save audio sample
        samples_dir = voice_bank_root / "audio_samples" / speaker_profile.key
        samples_dir.mkdir(parents=True, exist_ok=True)
        audio_sample_path = samples_dir / f"{source_episode}_{embedding_id_str}.wav"
        save_audio(trimmed, audio_sample_path, subtype="PCM_16", always_mono=True)

        # Add to database
        embedding_entry = voice_bank_manager.add_embedding(
            speaker_id=speaker_profile.id,
            embedding_path=embedding_path,
            source_episode=source_episode,
            source_timestamp=f"{start:.2f}-{end:.2f}",
            audio_sample_path=audio_sample_path,
            quality_score=quality_score,
        )

        # Enforce capacity limit
        existing_embeddings = voice_bank_manager.list_embeddings(speaker_profile.id)
        if len(existing_embeddings) > max_embeddings:
            # Remove lowest quality embeddings
            sorted_by_quality = sorted(
                existing_embeddings,
                key=lambda e: e.quality_score if e.quality_score is not None else 0.0,
            )
            to_remove = sorted_by_quality[: len(existing_embeddings) - max_embeddings]
            for entry_to_remove in to_remove:
                if entry_to_remove.id is not None:
                    voice_bank_manager.remove_embedding(entry_to_remove.id)
                try:
                    entry_to_remove.embedding_path.unlink(missing_ok=True)
                    if entry_to_remove.audio_sample_path:
                        entry_to_remove.audio_sample_path.unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass

        # Increment segment count
        voice_bank_manager.increment_segment_count(speaker_profile.key, increment=1)

        return IngestionResult(
            speaker_display_name=speaker_profile.display_name,
            start=start,
            end=end,
            duration=duration,
            accepted=True,
            embedding_id=embedding_entry.id,
            embedding_path=embedding_path,
            audio_sample_path=audio_sample_path,
            quality_score=quality_score,
        )

    except Exception as exc:
        LOGGER.error("Failed to ingest segment for %s: %s", speaker_profile.display_name, exc)
        return IngestionResult(
            speaker_display_name=speaker_profile.display_name,
            start=start,
            end=end,
            duration=duration,
            accepted=False,
            rejection_reason=f"Unexpected error: {exc}",
        )
