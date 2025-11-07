"""Audio input/output utility functions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # pragma: no cover - optional dependency
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("soundfile is required for audio I/O operations.") from exc

Waveform = NDArray[np.float32]
ResamplePoly = Callable[[Waveform, int, int], Waveform]

try:  # pragma: no cover - optional dependency
    from scipy.signal import resample_poly as _scipy_resample_poly
except ModuleNotFoundError:  # pragma: no cover - fallback
    resample_poly: ResamplePoly | None = None
else:
    resample_poly = cast(ResamplePoly, _scipy_resample_poly)


def _ensure_waveform(array: ArrayLike, *, copy: bool = False) -> Waveform:
    result = np.asarray(array, dtype=np.float32)
    if copy:
        result = result.copy()
    return result


__all__ = [
    "AudioClip",
    "compute_stats",
    "convert_format",
    "ensure_mono",
    "extract_segment",
    "load_audio",
    "resample_audio",
    "save_audio",
]


@dataclass(slots=True)
class AudioClip:
    """In-memory representation of an audio clip."""

    samples: NDArray[np.float32]
    sample_rate: int

    @property
    def channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.samples.size == 0 or self.sample_rate == 0:
            return 0.0
        total_samples = int(self.samples.shape[-1])
        return total_samples / float(self.sample_rate)


def load_audio(
    path: str | Path,
    *,
    target_sample_rate: int | None = None,
    mono: bool = True,
    dtype: str = "float32",
) -> AudioClip:
    """Load an audio clip from disk."""
    file_path = Path(path)
    data, sample_rate = sf.read(str(file_path), dtype=dtype, always_2d=False)
    waveform = _to_channel_first(_ensure_waveform(data))
    if mono:
        waveform = ensure_mono(waveform)
    if target_sample_rate and sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate, method="sinc_best")
        sample_rate = target_sample_rate
    return AudioClip(samples=waveform, sample_rate=sample_rate)


def save_audio(
    clip: AudioClip,
    path: str | Path,
    *,
    subtype: str = "PCM_16",
    always_mono: bool = False,
) -> None:
    """Write an audio clip to disk."""
    waveform = clip.samples
    if always_mono:
        waveform = ensure_mono(waveform)
    array = _to_soundfile_array(waveform)
    sf.write(str(path), array.T, clip.sample_rate, subtype=subtype)


def resample_audio(
    waveform: NDArray[np.float32],
    source_rate: int,
    target_rate: int,
    *,
    method: Literal["sinc_fast", "sinc_best", "fft"] = "sinc_best",
) -> NDArray[np.float32]:
    """Resample an audio waveform to a new sample rate."""
    if source_rate == target_rate:
        return _ensure_waveform(waveform, copy=True)

    if resample_poly is not None:
        ratio = Fraction(target_rate, source_rate).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator
        return _resample_poly(waveform, up, down)

    return _resample_fft(waveform, source_rate, target_rate, method=method)


def ensure_mono(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a waveform to mono by averaging channels if necessary."""
    if waveform.ndim == 1:
        return _ensure_waveform(waveform, copy=True)
    return _ensure_waveform(waveform.mean(axis=0, dtype=np.float32))


def extract_segment(
    clip: AudioClip,
    start_time: float,
    end_time: float,
) -> AudioClip:
    """Extract a slice from the audio clip between ``start_time`` and ``end_time``."""
    if start_time < 0 or end_time < 0 or end_time < start_time:
        raise ValueError("start_time must be <= end_time and both non-negative.")
    start_index = int(start_time * clip.sample_rate)
    end_index = int(end_time * clip.sample_rate)
    samples = clip.samples[..., start_index:end_index]
    return AudioClip(samples=samples, sample_rate=clip.sample_rate)


def convert_format(
    path: str | Path,
    output_path: str | Path,
    *,
    target_sample_rate: int | None = None,
    mono: bool = True,
    subtype: str = "PCM_16",
) -> AudioClip:
    """Load, optionally resample, and write audio to the requested format."""
    clip = load_audio(path, target_sample_rate=target_sample_rate, mono=mono)
    save_audio(clip, output_path, subtype=subtype, always_mono=mono)
    return clip


def compute_stats(clip: AudioClip) -> dict[str, float]:
    """Compute lightweight statistics for audio quality checks."""
    samples = clip.samples
    if samples.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak_dbfs": -math.inf,
        }
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    peak_dbfs = -math.inf if peak == 0 else 20 * math.log10(peak)
    return {
        "duration_seconds": clip.duration_seconds,
        "rms": rms,
        "peak_dbfs": peak_dbfs,
    }


def _to_channel_first(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Ensure arrays are channel-first for internal processing."""
    if arr.ndim == 1:
        return arr
    return arr.T


def _to_soundfile_array(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert channel-first arrays back to SoundFile's channel-last layout."""
    if waveform.ndim == 1:
        return waveform
    return waveform.T


def _resample_poly(waveform: Waveform, up: int, down: int) -> Waveform:
    """Resample using polyphase filtering when SciPy is available."""
    func = resample_poly
    assert func is not None, "resample_poly is expected to be available."
    if waveform.ndim == 1:
        return _ensure_waveform(func(waveform, up, down))

    channels: list[Waveform] = []
    for channel in waveform:
        channels.append(_ensure_waveform(func(channel, up, down)))
    stacked = np.vstack(channels)
    return _ensure_waveform(stacked)


def _resample_fft(
    waveform: Waveform,
    source_rate: int,
    target_rate: int,
    *,
    method: Literal["sinc_fast", "sinc_best", "fft"],
) -> Waveform:
    """Fallback resampling implementation using numpy FFT."""
    duration = waveform.shape[-1] / float(source_rate)
    target_samples = round(duration * target_rate)
    if waveform.ndim == 1:
        spectrum = np.fft.rfft(waveform)
        resampled = np.fft.irfft(spectrum, n=target_samples)
        return _ensure_waveform(resampled)

    channels: list[Waveform] = []
    for channel in waveform:
        spectrum = np.fft.rfft(channel)
        resampled = np.fft.irfft(spectrum, n=target_samples)
        channels.append(_ensure_waveform(resampled))
    return _ensure_waveform(np.vstack(channels))
