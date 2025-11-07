from __future__ import annotations

from pathlib import Path

from show_scribe.pipelines.audio_preprocessing import select_transcription_inputs
from show_scribe.exceptions import AudioPreprocessingCancelled


def test_audio_preprocessing_cancelled_class_importable() -> None:
    assert issubclass(AudioPreprocessingCancelled, Exception)


def test_select_transcription_inputs_prefers_enhanced(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    vocals = audio_dir / "audio_enhanced_vocals.wav"
    mix = audio_dir / "audio_enhanced_mix.wav"
    vocals.write_bytes(b"\x00\x00")
    mix.write_bytes(b"\x00\x00")

    selected_vocals, selected_mix = select_transcription_inputs(tmp_path)

    assert selected_vocals == vocals
    assert selected_mix == mix
