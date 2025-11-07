"""
Audio preprocessing pipeline for reality TV and noisy environments.

This module provides vocal separation and audio enhancement to improve
transcription accuracy in challenging acoustic conditions:
- Background music during dialogue
- Restaurant/crowd ambient noise
- Poor microphone placement
- Reverb and echo

Technologies:
- python-audio-separator (Demucs) for vocal extraction
- resemble-enhance for denoising and enhancement
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import librosa
import numpy as np
from show_scribe.audio_preprocess.clearervoice_wrapper import (
    ClearerVoiceUnavailable,
    ClearerVoiceWrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingArtifacts:
    """Materialised outputs from the preprocessing pipeline."""

    final_audio: Path
    vocals_audio: Path | None
    enhanced_audio: Path | None
    enhanced_mix_audio: Path | None
    report_path: Path
    report: Dict[str, Any]


class AudioPreprocessor:
    """Handles audio preprocessing for improved transcription accuracy."""

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dict with audio_preprocessing settings
        """
        raw_config = config.get("audio_preprocessing", {})
        self.config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
        self.master_setting = self.config.get("enable", False)

        raw_vocal_sep = self.config.get("vocal_separation", {})
        self.vocal_sep_config = dict(raw_vocal_sep) if isinstance(raw_vocal_sep, Mapping) else {}

        raw_enhancement = self.config.get("enhancement", {})
        self.enhancement_config = (
            dict(raw_enhancement) if isinstance(raw_enhancement, Mapping) else {}
        )

        retain_setting = self.config.get("retain_intermediates", True)
        self.retain_intermediates = self._coerce_bool(retain_setting, default=True)
        self.enhancement_provider = (
            str(self.enhancement_config.get("provider", "resemble")).strip().lower() or "resemble"
        )

        raw_clearervoice = self.enhancement_config.get("clearervoice", {})
        self.clearervoice_config = (
            dict(raw_clearervoice) if isinstance(raw_clearervoice, Mapping) else {}
        )
        self._clearervoice_wrapper: ClearerVoiceWrapper | None = None

    def preprocess(
        self, audio_path: str, output_dir: str = "temp/processed"
    ) -> PreprocessingArtifacts:
        """
        Main preprocessing pipeline with smart detection.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory for processed outputs

        Returns:
            PreprocessingArtifacts describing generated assets and report.
        """
        source = Path(audio_path).expanduser().resolve()
        processed_root = Path(output_dir).expanduser().resolve()
        processed_root.mkdir(parents=True, exist_ok=True)

        outputs = self._prepare_output_paths(processed_root, source)
        report_path = outputs["report"]

        if self.master_setting is False:
            logger.info("Audio preprocessing disabled via configuration; skipping.")
            final_audio = self._materialise_final_audio(source, outputs["final"])
            report: Dict[str, Any] = {
                "input_path": str(source),
                "output_path": str(final_audio),
                "analysis": None,
                "steps_applied": [],
                "preprocessing_enabled": False,
                "reason": "disabled",
                "files": {
                    "processed": str(final_audio),
                    "vocals": None,
                    "enhanced": None,
                    "enhanced_vocals": None,
                    "enhanced_mix": None,
                    "report": str(report_path),
                },
                "cleanup": {
                    "intermediates_retained": self.retain_intermediates,
                    "intermediates_purged": False,
                },
            }
            self._write_report(report_path, report)
            return PreprocessingArtifacts(
                final_audio=final_audio,
                vocals_audio=None,
                enhanced_audio=None,
                enhanced_mix_audio=None,
                report_path=report_path,
                report=report,
            )

        logger.info("Starting audio preprocessing for %s", source)

        analysis = self.analyze_audio_quality(str(source))
        logger.info(
            "Audio analysis: SNR=%.1fdB, Music=%.1f%%, Speech clarity=%.2f",
            analysis["snr"],
            analysis["music_ratio"] * 100.0,
            analysis["speech_clarity"],
        )

        steps_applied: list[str] = []
        timings_seconds: Dict[str, float] = {}
        warnings: list[str] = []

        current_audio = source
        vocals_path: Path | None = None
        enhanced_path: Path | None = None
        enhanced_mix_path: Path | None = None
        intermediates_purged = False

        if self._should_separate_vocals(analysis):
            logger.info("Applying vocal separation...")
            start = time.perf_counter()
            try:
                vocals_candidate = self.separate_vocals(current_audio, outputs["vocals"])
            except (
                Exception
            ) as exc:  # pragma: no cover - defensive guard around optional dependency
                logger.error("Vocal separation failed: %s", exc)
                vocals_candidate = None
                warnings.append(f"vocal_separation_failed: {exc}")
            timings_seconds["vocal_separation"] = time.perf_counter() - start

            if vocals_candidate is not None:
                vocals_path = vocals_candidate
                current_audio = vocals_candidate
                steps_applied.append("vocal_separation")
                logger.info("Vocal separation output: %s", vocals_candidate)
            else:
                logger.info("Skipping vocal separation due to error or missing dependencies.")
                with suppress(FileNotFoundError):
                    outputs["vocals"].unlink()
        else:
            logger.info("Skipping vocal separation (not needed).")

        if self._should_enhance(analysis):
            logger.info("Applying audio enhancement...")
            start = time.perf_counter()
            try:
                if self.enhancement_provider == "clearervoice":
                    enhanced_candidate, enhanced_mix_candidate = self._enhance_with_clearervoice(
                        current_audio=current_audio,
                        outputs=outputs,
                        source_audio=source,
                        vocals_path=vocals_path,
                    )
                else:
                    enhanced_candidate = self.enhance_audio(
                        current_audio, outputs["enhanced_vocals"]
                    )
                    enhanced_mix_candidate = None
            except (
                Exception
            ) as exc:  # pragma: no cover - defensive guard around optional dependency
                logger.error("Enhancement failed: %s", exc)
                enhanced_candidate = None
                enhanced_mix_candidate = None
                warnings.append(f"enhancement_failed: {exc}")
            timings_seconds["enhancement"] = time.perf_counter() - start

            if enhanced_candidate is not None:
                enhanced_path = enhanced_candidate
                current_audio = enhanced_candidate
                steps_applied.append("enhancement")
                logger.info("Audio enhancement output: %s", enhanced_candidate)
            else:
                logger.info("Skipping enhancement due to error or missing dependencies.")
                with suppress(FileNotFoundError):
                    outputs["enhanced_vocals"].unlink()

            if enhanced_mix_candidate is not None:
                enhanced_mix_path = enhanced_mix_candidate
                logger.info("Enhanced mix output: %s", enhanced_mix_candidate)
            elif self.enhancement_provider == "clearervoice":
                logger.info("Enhanced mix not produced; check ClearerVoice logs for details.")
                with suppress(FileNotFoundError):
                    outputs["enhanced_mix"].unlink()
        else:
            logger.info("Skipping enhancement (not needed).")

        final_audio = self._materialise_final_audio(current_audio, outputs["final"])

        if not self.retain_intermediates:
            if any(
                candidate is not None
                for candidate in (vocals_path, enhanced_path, enhanced_mix_path)
            ):
                intermediates_purged = True
            self._purge_intermediate_files(vocals_path, enhanced_path, enhanced_mix_path)
            vocals_path = None
            enhanced_path = None
            enhanced_mix_path = None

        report = {
            "input_path": str(source),
            "output_path": str(final_audio),
            "analysis": analysis,
            "steps_applied": steps_applied,
            "preprocessing_enabled": bool(steps_applied),
            "timings_seconds": timings_seconds,
            "warnings": warnings,
            "files": {
                "processed": str(final_audio),
                "vocals": str(vocals_path) if vocals_path else None,
                "enhanced": str(enhanced_path) if enhanced_path else None,
                "enhanced_vocals": str(enhanced_path) if enhanced_path else None,
                "enhanced_mix": str(enhanced_mix_path) if enhanced_mix_path else None,
                "report": str(report_path),
            },
            "cleanup": {
                "intermediates_retained": self.retain_intermediates,
                "intermediates_purged": intermediates_purged,
            },
        }

        self._write_report(report_path, report)

        return PreprocessingArtifacts(
            final_audio=final_audio,
            vocals_audio=vocals_path,
            enhanced_audio=enhanced_path,
            enhanced_mix_audio=enhanced_mix_path,
            report_path=report_path,
            report=report,
        )

    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """
        Analyze audio to determine if preprocessing is needed.

        Checks:
        - SNR (signal-to-noise ratio)
        - Music content ratio
        - Speech clarity
        - Reverb level

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with quality metrics
        """
        logger.info("Analyzing audio quality...")

        # Load first 60 seconds for analysis
        y, sr = librosa.load(audio_path, sr=16000, duration=60, mono=True)

        # 1. Calculate SNR
        rms = librosa.feature.rms(y=y)[0]
        snr = 20 * np.log10(np.max(rms) / (np.median(rms) + 1e-10))

        # 2. Detect music content (spectral flatness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        # Music has low flatness (tonal), noise has high flatness
        music_ratio = np.mean(spectral_flatness < 0.1)

        # 3. Speech clarity (zero-crossing rate)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        speech_clarity = 1 - np.std(zcr) / (np.mean(zcr) + 1e-10)
        speech_clarity = np.clip(speech_clarity, 0, 1)

        # 4. Reverb detection (spectral centroid variance)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        reverb_score = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10)
        reverb_score = np.clip(reverb_score / 0.5, 0, 1)  # Normalize to 0-1

        return {
            "snr": float(snr),
            "music_ratio": float(music_ratio),
            "speech_clarity": float(speech_clarity),
            "reverb_score": float(reverb_score),
            "duration_analyzed": len(y) / sr,
        }

    def _should_separate_vocals(self, analysis: Dict) -> bool:
        """
        Determine if vocal separation is needed.

        Args:
            analysis: Audio quality analysis dict

        Returns:
            True if vocal separation should be applied
        """
        # User explicitly enabled it
        enable_setting = self.vocal_sep_config.get("enable")
        if enable_setting is True:
            return True
        if enable_setting is False:
            return False

        # Auto-detect (enable_setting == "auto" or None)
        # High music content - definitely need separation
        if analysis["music_ratio"] > 0.3:  # >30% music
            logger.info(
                f"Auto-enabling vocal separation (music ratio: {analysis['music_ratio']:.1%})"
            )
            return True

        # Low speech clarity - might be music interfering
        if analysis["speech_clarity"] < 0.6:
            logger.info(
                f"Auto-enabling vocal separation (low speech clarity: {analysis['speech_clarity']:.2f})"
            )
            return True

        return False

    def _should_enhance(self, analysis: Dict) -> bool:
        """
        Determine if enhancement is needed.

        Args:
            analysis: Audio quality analysis dict

        Returns:
            True if enhancement should be applied
        """
        # User explicitly enabled it
        enable_setting = self.enhancement_config.get("enable")
        if enable_setting is True:
            return True
        if enable_setting is False:
            return False

        # Auto-detect (enable_setting == "auto" or None)
        # Low SNR - noisy audio
        if analysis["snr"] < 15:  # Less than 15 dB SNR
            logger.info(f"Auto-enabling enhancement (low SNR: {analysis['snr']:.1f}dB)")
            return True

        # High reverb
        if analysis["reverb_score"] > 0.5:
            logger.info(f"Auto-enabling enhancement (high reverb: {analysis['reverb_score']:.2f})")
            return True

        return False

    def separate_vocals(self, audio_path: Path, destination: Path) -> Path | None:
        """
        Extract vocals from audio using Demucs model.

        Removes background music, ambient noise, and other non-speech audio.

        Args:
            audio_path: Path to input audio
            output_dir: Directory for output files

        Returns:
            Path to vocals-only audio file
        """
        if self.enhancement_provider == "clearervoice":
            try:
                wrapper = self._get_clearervoice_wrapper()
            except RuntimeError as exc:
                logger.error("ClearerVoice not available for separation: %s", exc)
                return None
            return wrapper.separate_vocals(audio_path, destination)

        try:
            from audio_separator.separator import Separator
        except ImportError:
            logger.error("audio-separator not installed. Install with: pip install audio-separator")
            return None

        model = self.vocal_sep_config.get("model", "htdemucs")
        logger.info("Using vocal separation model: %s", model)

        # Initialize separator
        temp_dir = destination.parent / "_separator_outputs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        separator = Separator(
            log_level=logging.WARNING,  # Reduce noise
            model_file_dir=str(self.vocal_sep_config.get("model_dir", "models/audio_separation")),
            output_dir=str(temp_dir),
        )

        # Load model
        separator.load_model(model_filename=f"{model}.yaml")

        # Separate
        output_files = separator.separate(str(audio_path))

        vocals_file = self._select_vocals_file(output_files, temp_dir)

        if vocals_file is None:
            logger.warning("Vocals file not found in output, using original audio.")
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vocals_file, destination)
        with suppress(Exception):
            shutil.rmtree(temp_dir)
        return destination

    def enhance_audio(self, audio_path: Path, destination: Path) -> Path | None:
        """
        Enhance audio quality using Resemble-Enhance.

        Applies denoising and dereverberation to improve clarity.

        Args:
            audio_path: Path to input audio
            output_dir: Directory for output files

        Returns:
            Path to enhanced audio file
        """
        try:
            import torch
            import torchaudio
            from resemble_enhance.enhancer.inference import enhance
        except ImportError:
            logger.error(
                "resemble-enhance not installed. Install with: pip install resemble-enhance"
            )
            return None

        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        wav, sr = torchaudio.load(str(audio_path))

        # Resample to 16kHz if needed (Whisper's native rate)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000

        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Enhance
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Running enhancement on %s", device)

        enhanced_wav = enhance(
            wav,
            sr,
            device=device,
            nfe=self.enhancement_config.get("nfe", 64),  # Quality vs speed tradeoff
            solver=self.enhancement_config.get("solver", "midpoint"),
            lambd=self.enhancement_config.get("lambd", 0.5),  # Denoising strength
            tau=self.enhancement_config.get("tau", 0.5),  # Dereverberation strength
            denoise=bool(self.enhancement_config.get("denoise", True)),
            dereverb=bool(self.enhancement_config.get("dereverb", True)),
        )

        # Save
        torchaudio.save(str(destination), enhanced_wav.cpu(), sr)
        logger.info("Enhanced audio saved to %s", destination)

        return destination

    def _get_clearervoice_wrapper(self) -> ClearerVoiceWrapper:
        """Return a lazily-instantiated ClearerVoice wrapper."""
        if self._clearervoice_wrapper is not None:
            return self._clearervoice_wrapper

        cfg = self.clearervoice_config
        separation_model = str(cfg.get("separation_model", "MossFormer2_SS_16K"))
        enhancement_model = str(cfg.get("enhancement_model", "FRCRN_SE_16K"))
        super_resolution_model_raw = cfg.get("super_resolution_model")
        super_resolution_model = (
            str(super_resolution_model_raw).strip() or None
            if isinstance(super_resolution_model_raw, str)
            else None
        )

        try:
            target_sample_rate = int(
                cfg.get(
                    "target_sample_rate",
                    self.enhancement_config.get("target_sample_rate", 16_000),
                )
            )
        except (TypeError, ValueError):
            target_sample_rate = 16_000

        try:
            self._clearervoice_wrapper = ClearerVoiceWrapper(
                separation_model=separation_model,
                enhancement_model=enhancement_model,
                super_resolution_model=super_resolution_model,
                target_sample_rate=target_sample_rate,
                settings=cfg,
            )
        except ClearerVoiceUnavailable as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(str(exc)) from exc

        return self._clearervoice_wrapper

    def _enhance_with_clearervoice(
        self,
        *,
        current_audio: Path,
        outputs: Dict[str, Path],
        source_audio: Path,
        vocals_path: Path | None,
    ) -> Tuple[Path | None, Path | None]:
        """Run ClearerVoice enhancement for vocals and full mix."""
        wrapper = self._get_clearervoice_wrapper()

        vocals_source = vocals_path if vocals_path and vocals_path.exists() else current_audio
        if not vocals_source.exists():
            vocals_source = source_audio

        enhanced_vocals = wrapper.enhance_vocals(vocals_source, outputs["enhanced_vocals"])
        enhanced_mix = wrapper.enhance_mix(source_audio, outputs["enhanced_mix"])

        return enhanced_vocals, enhanced_mix

    @staticmethod
    def _select_vocals_file(file_paths: Iterable[str], search_dir: Path) -> Path | None:
        """Best-effort selection of a vocals track from Demucs output."""
        for candidate in file_paths:
            path = Path(candidate)
            if "vocals" in path.stem.lower() and path.exists():
                return path
        for candidate in search_dir.rglob("*.wav"):
            if "vocals" in candidate.stem.lower():
                return candidate
        return None

    @staticmethod
    def _materialise_final_audio(source: Path, destination: Path) -> Path:
        """Copy ``source`` to ``destination`` if needed and return the resulting path."""
        source = source.expanduser().resolve()
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == destination:
            return destination
        try:
            shutil.copy2(source, destination)
        except shutil.SameFileError:
            pass
        return destination

    @staticmethod
    def _write_report(path: Path, payload: Dict[str, Any]) -> None:
        """Persist preprocessing metadata to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _prepare_output_paths(self, processed_root: Path, source: Path) -> Dict[str, Path]:
        """Initialise canonical output paths and remove any stale artifacts."""
        outputs = {
            "vocals": processed_root / "audio_vocals.wav",
            "enhanced_vocals": processed_root / "audio_enhanced_vocals.wav",
            "enhanced_mix": processed_root / "audio_enhanced_mix.wav",
            "final": processed_root / "audio_processed.wav",
            "report": processed_root / "preprocessing_report.json",
        }
        stale_paths = {
            processed_root / "audio_enhanced.wav",  # Legacy name
        }
        for path in (*outputs.values(), *stale_paths):
            if path == source:
                continue
            with suppress(FileNotFoundError):
                path.unlink()
        return outputs

    @staticmethod
    def _purge_intermediate_files(*paths: Path | None) -> None:
        """Remove intermediate artifacts when configured to purge."""
        for candidate in paths:
            if candidate is None:
                continue
            with suppress(OSError):
                candidate.unlink()

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        """Return a boolean flag from user-provided configuration values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default


def select_transcription_inputs(workdir: str | Path) -> tuple[Path, Path]:
    """
    Return preferred audio artifacts for transcription/diarization and bleep detection.

    Args:
        workdir: Episode directory containing audio artifacts.

    Returns:
        Tuple of (vocals_path, mix_path).

    Raises:
        FileNotFoundError: When the required artifacts are missing.
    """

    root = Path(workdir).expanduser().resolve()

    def _candidate_paths(filename: str) -> list[Path]:
        return [
            root / filename,
            root / "audio" / filename,
            root / "processed_audio" / filename,
        ]

    def _resolve(names: tuple[str, ...], label: str) -> Path:
        for name in names:
            for candidate in _candidate_paths(name):
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"{label} artifact not found under {root}. "
            "Run CREATE AUDIO to generate enhanced stems."
        )

    vocals = _resolve(
        (
            "audio_enhanced_vocals.wav",
            "audio_vocals.wav",
            "audio_processed.wav",
            "audio_extracted.wav",
        ),
        "Enhanced vocals",
    )
    mix = _resolve(
        ("audio_enhanced_mix.wav", "audio_processed.wav", "audio_extracted.wav"),
        "Enhanced mix",
    )
    return vocals, mix


def create_reality_tv_preset() -> Dict:
    """
    Create optimized preset for reality TV scenarios.

    Aggressive preprocessing for:
    - Restaurant scenes
    - Background music
    - Crowd noise
    - Poor mic placement

    Returns:
        Configuration dict
    """
    return {
        "audio_preprocessing": {
            "enable": True,
            "retain_intermediates": True,
            "vocal_separation": {
                "enable": True,  # Always separate for reality TV
                "model": "htdemucs",  # Best quality
            },
            "enhancement": {
                "enable": True,  # Always enhance for reality TV
                "denoise": True,
                "dereverb": True,
                "nfe": 64,  # High quality
                "lambd": 0.7,  # Aggressive denoising
                "tau": 0.6,  # Moderate dereverberation
                "provider": "resemble",
                "clearervoice": {
                    "separation_model": "MossFormer2_SS_16K",
                    "enhancement_model": "FRCRN_SE_16K",
                    "target_sample_rate": 16_000,
                },
            },
        },
        "whisper": {
            "model": "large-v3",
            "beam_size": 10,  # Higher for better accuracy
            "temperature": 0,  # Deterministic
            "condition_on_previous_text": True,
            "initial_prompt": "This is reality TV dialogue in a noisy restaurant with background music. Multiple people are speaking.",
        },
    }


# Quick test/demo function
if __name__ == "__main__":
    import json

    # Example usage
    preset = create_reality_tv_preset()
    print("Reality TV Preset:")
    print(json.dumps(preset, indent=2))

    # Test with a file
    # preprocessor = AudioPreprocessor(preset)
    # processed_path, report = preprocessor.preprocess("test_audio.wav")
    # print(f"\nProcessed audio: {processed_path}")
    # print(f"Report: {json.dumps(report, indent=2)}")
