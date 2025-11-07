"""Helpers for generating waveform and spectrogram data for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    from scipy.signal import stft
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("scipy is required for spectrogram visualizations.") from exc

from .audio_io import AudioClip, ensure_mono

__all__ = ["SpectrogramData", "WaveformData", "compute_spectrogram", "compute_waveform"]


class WaveformData(NamedTuple):
    times: NDArray[np.float32]
    amplitudes: NDArray[np.float32]


@dataclass(slots=True)
class SpectrogramData:
    """Container for spectrogram matrices."""

    times: NDArray[np.float32]
    frequencies: NDArray[np.float32]
    magnitudes: NDArray[np.float32]


def compute_waveform(
    clip: AudioClip,
    *,
    max_points: int = 5_000,
) -> WaveformData:
    """Return a downsampled waveform suitable for visualization."""
    mono = ensure_mono(clip.samples)
    total_samples = mono.shape[-1]
    if total_samples == 0:
        return WaveformData(
            times=np.zeros(0, dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
        )

    step = max(1, total_samples // max_points)
    trimmed = mono[: step * (total_samples // step)]
    window = trimmed.reshape(-1, step)
    amplitudes = window.mean(axis=1).astype(np.float32)
    times = np.arange(amplitudes.shape[0], dtype=np.float32) * step / float(clip.sample_rate)
    return WaveformData(times=times, amplitudes=amplitudes)


def compute_spectrogram(
    clip: AudioClip,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectrogramData:
    """Compute a magnitude spectrogram."""
    mono = ensure_mono(clip.samples)
    if mono.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return SpectrogramData(
            times=empty,
            frequencies=empty,
            magnitudes=np.zeros((0, 0), dtype=np.float32),
        )

    frequencies, times, zxx = stft(
        mono,
        clip.sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
    )
    magnitudes = np.abs(zxx).astype(np.float32)
    return SpectrogramData(
        times=times.astype(np.float32),
        frequencies=frequencies.astype(np.float32),
        magnitudes=magnitudes,
    )
