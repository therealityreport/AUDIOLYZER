"""Tests for audio visualization helpers."""

from __future__ import annotations

import numpy as np

from show_scribe.utils.audio_io import AudioClip
from show_scribe.utils.audio_visualization import (
    SpectrogramData,
    WaveformData,
    compute_spectrogram,
    compute_waveform,
)


def make_clip(duration_seconds: float = 1.0, sample_rate: int = 16000) -> AudioClip:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return AudioClip(samples=waveform, sample_rate=sample_rate)


def test_compute_waveform_downsamples() -> None:
    clip = make_clip(duration_seconds=0.5)
    waveform = compute_waveform(clip, max_points=50)
    assert isinstance(waveform, WaveformData)
    assert waveform.times.size <= 50
    assert waveform.amplitudes.size == waveform.times.size


def test_compute_spectrogram_returns_data() -> None:
    clip = make_clip(duration_seconds=0.5)
    spectrogram = compute_spectrogram(clip, n_fft=256, hop_length=64)
    assert isinstance(spectrogram, SpectrogramData)
    assert spectrogram.magnitudes.ndim == 2
    assert spectrogram.times.size > 0
    assert spectrogram.frequencies.size > 0
