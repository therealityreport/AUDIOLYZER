"""Integration tests for the voice identification pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from show_scribe.pipelines.alignment.align_asr_diar import (
    AlignedSegment,
    AlignedWord,
    AlignmentMetadata,
    AlignmentResult,
)
from show_scribe.pipelines.diarization.pyannote_pipeline import (
    DiarizationMetadata,
    DiarizationResult,
    DiarizationSegment,
)
from show_scribe.pipelines.speaker_id.voice_bank import VoiceBankPipeline
from show_scribe.storage.db import SQLiteDatabase
from show_scribe.storage.paths import PathsConfig
from show_scribe.storage.voice_bank_manager import VoiceBankManager


class _MeanEmbeddingBackend:
    """Deterministic embedding backend for testing."""

    target_sample_rate = 16000

    def encode(self, clip) -> np.ndarray:  # type: ignore[override]
        samples = clip.samples
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        if samples.size == 0:
            return np.zeros(2, dtype=np.float32)
        mean_val = float(np.mean(samples))
        second_moment = float(np.mean(samples**2))
        return np.array([mean_val, second_moment], dtype=np.float32)


@pytest.mark.integration
def test_voice_bank_pipeline_matches_seeded_speakers(tmp_path: Path) -> None:
    paths = PathsConfig(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        cache_dir=tmp_path / "data" / "cache",
        temp_dir=tmp_path / "data" / "tmp",
        models_dir=tmp_path / "data" / "models",
        voice_bank_db=tmp_path / "data" / "voice_bank" / "voice.sqlite3",
        logs_dir=tmp_path / "logs",
    )
    paths.ensure_directories()

    config = {
        "voice_identification": {
            "embedding_provider": "pyannote",
            "high_confidence_threshold": 0.05,
            "medium_confidence_threshold": 0.05,
            "low_confidence_threshold": 0.05,
            "auto_register_new_speakers": False,
            "min_samples_required": 1,
        },
        "voice_bank": {
            "max_embeddings_per_speaker": 5,
            "min_sample_duration_seconds": 0.1,
        },
    }

    database = SQLiteDatabase(paths.voice_bank_db)
    database.initialize()
    manager = VoiceBankManager(database)

    alice = manager.upsert_speaker(display_name="Alice", key="alice")
    bob = manager.upsert_speaker(display_name="Bob", key="bob")
    assert alice.id is not None and bob.id is not None

    embedding_root = paths.data_root / "voice_bank" / "embeddings"
    (embedding_root / "alice").mkdir(parents=True, exist_ok=True)
    (embedding_root / "bob").mkdir(parents=True, exist_ok=True)

    alice_vector = np.array([0.1, 0.01], dtype=np.float32)
    bob_vector = np.array([0.2, 0.04], dtype=np.float32)

    alice_embedding_path = embedding_root / "alice" / "seed.npy"
    bob_embedding_path = embedding_root / "bob" / "seed.npy"
    np.save(alice_embedding_path, alice_vector)
    np.save(bob_embedding_path, bob_vector)

    manager.add_embedding(
        speaker_id=alice.id, embedding_path=alice_embedding_path, source_episode="seed"
    )
    manager.add_embedding(
        speaker_id=bob.id, embedding_path=bob_embedding_path, source_episode="seed"
    )

    sample_rate = 16000
    audio_samples = np.concatenate(
        [
            np.full(sample_rate, 0.1, dtype=np.float32),
            np.full(sample_rate, 0.2, dtype=np.float32),
        ]
    )
    audio_path = tmp_path / "episode.wav"
    sf.write(audio_path, audio_samples, sample_rate)

    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(segment_id=0, start=0.0, end=1.0, speaker="SPEAKER_00"),
            DiarizationSegment(segment_id=1, start=1.0, end=2.0, speaker="SPEAKER_01"),
        ],
        metadata=DiarizationMetadata(
            model="stub",
            speaker_count=2,
            duration=2.0,
            inference_seconds=None,
            parameters={},
        ),
    )

    alignment = AlignmentResult(
        segments=[
            AlignedSegment(
                segment_id=0,
                start=0.0,
                end=1.0,
                speaker="SPEAKER_00",
                text="Hello",
                words=[
                    AlignedWord(word="Hello", start=0.0, end=0.5, speaker="SPEAKER_00"),
                ],
            ),
            AlignedSegment(
                segment_id=1,
                start=1.0,
                end=2.0,
                speaker="SPEAKER_01",
                text="World",
                words=[
                    AlignedWord(word="World", start=1.0, end=1.5, speaker="SPEAKER_01"),
                ],
            ),
        ],
        metadata=AlignmentMetadata(
            asr_segment_count=2,
            diarization_segment_count=2,
            words_aligned_count=2,
            unaligned_word_count=0,
            speakers=["SPEAKER_00", "SPEAKER_01"],
        ),
    )

    pipeline = VoiceBankPipeline(
        config,
        paths,
        embedding_backend=_MeanEmbeddingBackend(),
        voice_bank=manager,
    )

    result = pipeline.identify(
        episode_id="TEST_S01E01",
        audio_path=audio_path,
        diarization=diarization,
        alignment=alignment,
    )

    assignments = {assignment.cluster_id: assignment for assignment in result.assignments}
    assert assignments["SPEAKER_00"].display_name == "Alice"
    assert assignments["SPEAKER_01"].display_name == "Bob"
    assert result.matched == 2
    assert result.unmatched == 0
