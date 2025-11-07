"""
Adapter around the ClearerVoice Studio toolkit used by the audio preprocessing stack.

The wrapper hides the fairly opinionated I/O model exposed by the upstream package and
provides simple helpers that operate on ``Path`` objects and in-memory numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Iterable, Mapping

import librosa
import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from clearvoice import ClearVoice
except ImportError:  # pragma: no cover - handled at runtime
    ClearVoice = None  # type: ignore[assignment]


class ClearerVoiceUnavailable(RuntimeError):
    """Raised when the ``clearvoice`` dependency is missing."""


@dataclass(slots=True)
class _SeparationCache:
    """Cached separation results to avoid recomputing ClearerVoice models."""

    vocals: np.ndarray
    stems: tuple[np.ndarray, ...]
    sample_rate: int


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Return a contiguous mono waveform."""
    if waveform.ndim == 1:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[0] == 1:
            return np.ascontiguousarray(waveform[0], dtype=np.float32)
        if waveform.shape[1] == 1:
            return np.ascontiguousarray(waveform[:, 0], dtype=np.float32)
        return np.ascontiguousarray(np.mean(waveform, axis=0), dtype=np.float32)
    raise ValueError(f"Unsupported waveform shape: {waveform.shape!r}")


def _ensure_unit_range(waveform: np.ndarray) -> np.ndarray:
    """Clip the waveform to the [-1, 1] range."""
    return np.clip(waveform, -1.0, 1.0, out=waveform)


class ClearerVoiceWrapper:
    """Thin adapter around ClearerVoice Studio models."""

    def __init__(
        self,
        *,
        separation_model: str = "MossFormer2_SS_16K",
        enhancement_model: str = "FRCRN_SE_16K",
        super_resolution_model: str | None = None,
        target_sample_rate: int = 16_000,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        if ClearVoice is None:  # pragma: no cover - exercised when dependency missing
            raise ClearerVoiceUnavailable(
                "clearvoice package is not installed. "
                "Install with `pip install clearvoice` to enable ClearerVoice integration."
            )

        self._separation_model_name = separation_model
        self._enhancement_model_name = enhancement_model
        self._super_resolution_model_name = super_resolution_model
        self._target_sample_rate = target_sample_rate
        self._settings = dict(settings or {})

        self._separation_model: ClearVoice | None = None
        self._enhancement_model: ClearVoice | None = None
        self._super_resolution_model: ClearVoice | None = None

        self._separation_cache: dict[Path, _SeparationCache] = {}

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def separate_vocals(self, audio_path: Path, destination: Path) -> Path | None:
        """Run ClearerVoice separation and persist the dominant vocal stem."""
        vocals, sample_rate = self._extract_vocals(audio_path)
        if vocals is None:
            return None

        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), vocals, sample_rate)
        self._register_cache(destination, vocals, sample_rate)
        return destination

    def enhance_vocals(self, audio_path: Path, destination: Path) -> Path | None:
        """Enhance a vocal-focused track."""
        waveform, sample_rate = self._load_waveform(audio_path)
        if waveform is None:
            return None

        enhanced = self._enhance_waveform(waveform)
        if enhanced is None:
            return None

        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), enhanced, sample_rate)
        self._register_cache(destination, enhanced, sample_rate)
        return destination

    def enhance_mix(self, audio_path: Path, destination: Path) -> Path | None:
        """Enhance the full mix while preserving ambience."""
        waveform, sample_rate = self._load_waveform(audio_path)
        if waveform is None:
            return None

        enhanced = self._enhance_waveform(waveform)
        if enhanced is None:
            return None

        if self._super_resolution_model_name:
            enhanced = self._apply_super_resolution(enhanced)

        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), enhanced, sample_rate)
        self._register_cache(destination, enhanced, sample_rate)
        return destination

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_vocals(self, audio_path: Path) -> tuple[np.ndarray | None, int]:
        key = audio_path.expanduser().resolve()
        cached = self._separation_cache.get(key)
        if cached is not None:
            return cached.vocals.copy(), cached.sample_rate

        waveform, sample_rate = self._load_waveform(audio_path)
        if waveform is None:
            return None, self._target_sample_rate

        stems = self._run_separation(audio_path)
        if not stems:
            LOGGER.warning("ClearerVoice separation produced no stems for %s", audio_path)
            return None, sample_rate

        vocals = self._select_vocal_stem(stems, sample_rate)
        vocals = _ensure_unit_range(vocals)
        self._register_cache(key, vocals, sample_rate, stems=tuple(stems))
        return vocals.copy(), sample_rate

    def _load_waveform(self, audio_path: Path) -> tuple[np.ndarray | None, int]:
        try:
            data, sample_rate = sf.read(str(audio_path), always_2d=False)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to read audio at %s: %s", audio_path, exc)
            return None, self._target_sample_rate

        mono = _to_mono(np.asarray(data))
        if sample_rate != self._target_sample_rate:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=self._target_sample_rate)
            sample_rate = self._target_sample_rate
        return mono, sample_rate

    def _run_separation(self, audio_path: Path) -> list[np.ndarray]:
        if self._separation_model is None:
            LOGGER.info("Loading ClearerVoice separation model: %s", self._separation_model_name)
            self._separation_model = ClearVoice(
                task="speech_separation",
                model_names=[self._separation_model_name],
            )

        result = self._separation_model(str(audio_path))
        if isinstance(result, list):
            stems = [self._normalize_stem(stem) for stem in result]
        elif isinstance(result, np.ndarray):
            stems = [self._normalize_stem(result)]
        elif isinstance(result, dict):
            stems = [self._normalize_stem(value) for value in result.values()]
        else:  # pragma: no cover - defensive
            LOGGER.warning("Unexpected separation output type %s", type(result))
            stems = []
        return stems

    def _enhance_waveform(self, waveform: np.ndarray) -> np.ndarray | None:
        if self._enhancement_model is None:
            LOGGER.info("Loading ClearerVoice enhancement model: %s", self._enhancement_model_name)
            self._enhancement_model = ClearVoice(
                task="speech_enhancement",
                model_names=[self._enhancement_model_name],
            )

        tensor = np.asarray(waveform, dtype=np.float32)
        if tensor.ndim == 1:
            tensor = tensor[np.newaxis, :]
        enhanced = self._enhancement_model.call_t2t_mode(tensor)
        if enhanced is None:
            LOGGER.error("ClearerVoice enhancement returned no data.")
            return None
        enhanced = _to_mono(np.asarray(enhanced))
        return _ensure_unit_range(enhanced)

    def _apply_super_resolution(self, waveform: np.ndarray) -> np.ndarray:
        if self._super_resolution_model_name is None:
            return waveform
        if self._super_resolution_model is None:
            LOGGER.info(
                "Loading ClearerVoice super-resolution model: %s", self._super_resolution_model_name
            )
            self._super_resolution_model = ClearVoice(
                task="speech_enhancement",
                model_names=[self._super_resolution_model_name],
            )
        tensor = waveform[np.newaxis, :]
        enhanced = self._super_resolution_model.call_t2t_mode(tensor)
        if enhanced is None:  # pragma: no cover - defensive
            return waveform
        enhanced = _to_mono(np.asarray(enhanced))
        return _ensure_unit_range(enhanced)

    def _normalize_stem(self, stem: np.ndarray) -> np.ndarray:
        normalized = _to_mono(np.asarray(stem, dtype=np.float32))
        max_val = np.max(np.abs(normalized)) or 1.0
        if not math.isfinite(max_val):
            return normalized
        return normalized / max_val

    def _select_vocal_stem(self, stems: Iterable[np.ndarray], sample_rate: int) -> np.ndarray:
        """Choose the most speech-like stem using a spectral speechiness heuristic."""
        best_index = 0
        best_score = -float("inf")
        stems_list = list(stems)

        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        speech_band = (freqs >= 120.0) & (freqs <= 4000.0)

        for idx, stem in enumerate(stems_list):
            mono = _to_mono(stem)
            spec = np.abs(librosa.stft(mono, n_fft=1024, hop_length=256, win_length=1024))
            if spec.size == 0:
                continue

            voice_energy = float(np.mean(spec[speech_band])) if speech_band.any() else 0.0
            broadband_energy = float(np.mean(spec) + 1e-9)
            zcr = float(
                np.mean(librosa.feature.zero_crossing_rate(mono, frame_length=1024, hop_length=256))
            )

            score = (voice_energy / broadband_energy) - 0.5 * zcr
            if score > best_score:
                best_score = score
                best_index = idx

        return _to_mono(stems_list[best_index])

    def _register_cache(
        self,
        key: Path,
        vocals: np.ndarray,
        sample_rate: int,
        *,
        stems: tuple[np.ndarray, ...] | None = None,
    ) -> None:
        cache_entry = _SeparationCache(
            vocals=_to_mono(np.asarray(vocals, dtype=np.float32)),
            stems=stems or tuple(),
            sample_rate=sample_rate,
        )
        self._separation_cache[key] = cache_entry
