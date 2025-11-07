"""Audio extraction and enhancement helpers for UI-triggered workflows."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which

LOGGER = logging.getLogger(__name__)

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_ACTIVE_AUDIO_PROCESSES: set[subprocess.Popen[str]] = set()
_AUDIO_PROCESS_LOCK = threading.Lock()
_AUDIO_CANCEL_EVENT = threading.Event()
_FFMPEG_PATH: str | None = None


class AudioPreprocessingError(RuntimeError):
    """Raised when an external preprocessing command fails."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str | None = None,
        artifacts: dict[str, Path | None] | None = None,
    ) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.artifacts: dict[str, Path | None] = artifacts or {}


class AudioPreprocessingCancelled(AudioPreprocessingError):
    """Raised when audio preprocessing is stopped by an explicit user request."""


def _register_audio_process(process: subprocess.Popen[str]) -> None:
    with _AUDIO_PROCESS_LOCK:
        _ACTIVE_AUDIO_PROCESSES.add(process)


def _unregister_audio_process(process: subprocess.Popen[str]) -> None:
    with _AUDIO_PROCESS_LOCK:
        _ACTIVE_AUDIO_PROCESSES.discard(process)


def reset_audio_cancellation() -> None:
    """Clear any outstanding cancellation request before starting a new run."""

    _AUDIO_CANCEL_EVENT.clear()


def request_audio_cancellation() -> bool:
    """Signal any active audio preprocessing commands to terminate.

    Returns ``True`` when at least one running process was signalled.
    """

    _AUDIO_CANCEL_EVENT.set()

    with _AUDIO_PROCESS_LOCK:
        processes = list(_ACTIVE_AUDIO_PROCESSES)

    if not processes:
        return False

    any_signalled = False
    for process in processes:
        if process.poll() is not None:
            continue
        any_signalled = True
        with contextlib.suppress(OSError, ProcessLookupError):
            process.terminate()

    # Give processes a short grace period to exit cleanly before escalating.
    deadline = time.time() + 5.0
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining or 0.1)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                process.kill()

    return any_signalled


def _raise_if_cancelled(stage: str) -> None:
    if _AUDIO_CANCEL_EVENT.is_set():
        raise AudioPreprocessingCancelled(
            f"{stage} cancelled by user request.",
            stderr="CREATE AUDIO run cancelled by user request.",
        )


def ensure_dir(path: Path) -> None:
    """Ensure parent directories exist for the provided path."""
    path.mkdir(parents=True, exist_ok=True)


def _find_ffmpeg() -> str | None:
    """Find ffmpeg executable in PATH or common installation locations."""

    # Try PATH first
    ffmpeg_in_path = which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path

    # Try common macOS Homebrew locations
    common_locations = [
        "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",  # Intel Homebrew
    ]

    for location in common_locations:
        if Path(location).exists():
            return location

    return None


def ensure_ffmpeg_available() -> str:
    """Ensure ffmpeg is available and return its path.

    Raises RuntimeError if ffmpeg cannot be found.
    """
    global _FFMPEG_PATH

    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = _find_ffmpeg()

    if _FFMPEG_PATH is None:
        raise RuntimeError(
            "ffmpeg is required for audio preprocessing but was not found in PATH. "
            "Install ffmpeg and ensure it is accessible before rerunning."
        )

    return _FFMPEG_PATH


def ffmpeg_extract_wav(source_media: Path, out_wav: Path) -> None:
    """Extract mono 16 kHz WAV audio from the provided media file."""

    ffmpeg_path = ensure_ffmpeg_available()
    ensure_dir(out_wav.parent)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(out_wav),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - delegated failure path
        raise AudioPreprocessingError(
            f"ffmpeg failed while extracting audio (exit code {exc.returncode}).",
            stderr=exc.stderr,
        ) from exc


def _get_python_minor_version(executable: Path) -> tuple[int, int] | None:
    """Return the (major, minor) version tuple for a Python executable or ``None`` on failure."""

    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    version = result.stdout.strip()
    if not version:
        return None

    try:
        major_str, minor_str = version.split(".", 1)
        return int(major_str), int(minor_str)
    except ValueError:
        return None


def _select_preprocessing_python() -> str:
    """Pick a Python 3.11 interpreter when the host process runs on 3.13+ (deepspeed incompatibility)."""

    if sys.version_info < (3, 13):
        return sys.executable

    current = Path(sys.executable)
    search_roots = {current.parent, Path(sys.prefix) / "bin"}
    candidate_names = ("python3.11", "python3", "python")

    for root in search_roots:
        for name in candidate_names:
            candidate = root / name
            if not candidate.exists():
                continue

            version = _get_python_minor_version(candidate)
            if version == (3, 11):
                LOGGER.warning("Using Python 3.11 for preprocessing: %s", candidate)
                return str(candidate)

    LOGGER.warning(
        "Python 3.11 interpreter not found; continuing with current interpreter %s which may fail.",
        sys.executable,
    )
    return sys.executable


def _run_checked(cmd: list[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and normalise failures into ``AudioPreprocessingError``."""

    _raise_if_cancelled(stage)

    # Set PyTorch MPS fallback for Apple Silicon compatibility
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except OSError as exc:  # pragma: no cover - delegated failure path
        raise AudioPreprocessingError(
            f"{stage} failed to start. Command: {' '.join(cmd)}. Error: {exc}"
        ) from exc

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    _register_audio_process(process)
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                stdout_chunks.append(stdout or "")
                stderr_chunks.append(stderr or "")
                break
            except subprocess.TimeoutExpired:
                if _AUDIO_CANCEL_EVENT.is_set():
                    with contextlib.suppress(OSError, ProcessLookupError):
                        process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(OSError, ProcessLookupError):
                            process.kill()
                    raise AudioPreprocessingCancelled(
                        f"{stage} cancelled by user request.",
                        stderr="CREATE AUDIO run cancelled by user request.",
                    )
                continue
    finally:
        _unregister_audio_process(process)

    returncode = process.returncode
    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    if returncode != 0:  # pragma: no cover - delegated failure path
        LOGGER.error(f"{stage} failed with exit code {returncode}")
        LOGGER.error(f"{stage} stderr: {stderr_text}")
        LOGGER.error(f"{stage} stdout: {stdout_text}")
        LOGGER.error(f"{stage} command: {' '.join(cmd)}")
        raise AudioPreprocessingError(
            f"{stage} failed (exit code {returncode}). "
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {stderr_text}\n"
            f"stdout: {stdout_text}",
            stderr=stderr_text,
        )

    if stdout_text:
        LOGGER.info(f"{stage} stdout: {stdout_text}")

    return subprocess.CompletedProcess(cmd, returncode, stdout_text, stderr_text)


def run_audio_preprocessing(
    in_wav: Path,
    out_vocals: Path,
    out_enhanced_vocals: Path,
    out_enhanced_mix: Path,
    preset_yaml: Path,
) -> None:
    """Execute preprocessing steps:

    1) Separate vocals from mix (if enabled)
    2) Enhance the isolated vocals (if enabled)
    3) Enhance the original mix (if enabled)

    Respects enable flags in preset config for vocal_separation and enhancement.
    """

    ensure_dir(out_vocals.parent)

    # Load preset config to check enable flags
    import yaml

    try:
        with open(preset_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        LOGGER.warning("Could not load preset config, assuming all steps enabled: %s", exc)
        config = {}

    audio_preproc = config.get("audio_preprocessing", {})
    vocal_sep_config = audio_preproc.get("vocal_separation", {})
    enhancement_config = audio_preproc.get("enhancement", {})

    separation_enabled = vocal_sep_config.get("enable", True)
    enhancement_enabled = enhancement_config.get("enable", True)

    # Force Python 3.11 for preprocessing (deepspeed/resemble-enhance incompatible with 3.13+)
    python_exe = _select_preprocessing_python()

    # Step 1: Vocal separation (if enabled)
    if separation_enabled:
        LOGGER.info("Running vocal separation...")
        # Create stderr log path for debugging
        stderr_log = out_vocals.parent / "reports" / "preproc_separate_stderr.log"
        stderr_log.parent.mkdir(parents=True, exist_ok=True)

        _run_checked(
            [
                python_exe,
                "-m",
                "preproc.separate",
                "--preset",
                str(preset_yaml),
                "--input",
                str(in_wav),
                "--output",
                str(out_vocals),
                "--stderr-log",
                str(stderr_log),
            ],
            stage="Vocal separation",
        )

        # CRITICAL: Resample Demucs output to 16kHz mono for Whisper compatibility
        # Demucs outputs 44.1kHz stereo by default, which causes Whisper to hang
        LOGGER.info("Resampling vocals to 16kHz mono for Whisper compatibility...")
        import soundfile as sf
        import resampy

        y, sr = sf.read(str(out_vocals))
        # Convert stereo to mono if needed
        if y.ndim == 2:
            y = y.mean(axis=1)
        # Resample if needed
        if sr != 16000:
            y = resampy.resample(y, sr, 16000, filter="kaiser_best")
        # Write resampled audio
        sf.write(str(out_vocals), y, 16000, subtype="PCM_16")
        LOGGER.info(
            f"Resampled vocals: {sr}Hz -> 16000Hz, channels: {2 if y.ndim == 2 else 1} -> 1"
        )
    else:
        LOGGER.info("Vocal separation disabled, copying raw audio to vocals output")
        import shutil

        shutil.copy2(str(in_wav), str(out_vocals))

    # Step 2 & 3: Enhancement (if enabled)
    enhancement_succeeded = False
    enhancement_skip_reason = None

    if enhancement_enabled:
        try:
            LOGGER.info("Running enhancement on vocals...")
            _run_checked(
                [
                    python_exe,
                    "-m",
                    "preproc.enhance",
                    "--preset",
                    str(preset_yaml),
                    "--input",
                    str(out_vocals),
                    "--output",
                    str(out_enhanced_vocals),
                ],
                stage="Enhance isolated vocals",
            )

            LOGGER.info("Running enhancement on full mix...")
            _run_checked(
                [
                    python_exe,
                    "-m",
                    "preproc.enhance",
                    "--preset",
                    str(preset_yaml),
                    "--input",
                    str(in_wav),
                    "--output",
                    str(out_enhanced_mix),
                ],
                stage="Enhance full mix",
            )
            enhancement_succeeded = True
            LOGGER.info("Enhancement completed successfully")
        except AudioPreprocessingError as exc:
            LOGGER.warning(f"Enhancement failed: {exc}. Falling back to unenhanced audio.")
            enhancement_skip_reason = f"failed: {str(exc)[:200]}"
            # Fall through to copy fallback files
        except Exception as exc:
            LOGGER.warning(
                f"Enhancement failed with unexpected error: {exc}. Falling back to unenhanced audio."
            )
            enhancement_skip_reason = f"failed: {str(exc)[:200]}"
            # Fall through to copy fallback files

    if not enhancement_succeeded:
        if enhancement_skip_reason is None:
            enhancement_skip_reason = "disabled" if not enhancement_enabled else "skipped"
        LOGGER.info(
            f"Enhancement {enhancement_skip_reason}, copying vocals/extracted to enhanced outputs"
        )
        import shutil

        shutil.copy2(str(out_vocals), str(out_enhanced_vocals))
        shutil.copy2(str(in_wav), str(out_enhanced_mix))

    # Write preprocessing report
    report_path = out_enhanced_vocals.parent / "reports" / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "vocal_separation": {"enabled": separation_enabled},
        "enhancement": {
            "enabled": enhancement_enabled,
            "succeeded": enhancement_succeeded,
            "reason": enhancement_skip_reason,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def extract_and_enhance(
    media_path: Path, episode_dir: Path, preset_yaml: Path
) -> dict[str, Path | None]:
    """Produce extracted, vocals, enhanced-vocals, and enhanced-mix audio artifacts for an episode."""

    ensure_ffmpeg_available()

    episode_dir = episode_dir.expanduser().resolve()
    ensure_dir(episode_dir)

    media_path = media_path.expanduser().resolve()
    preset_yaml = preset_yaml.expanduser().resolve()
    if not preset_yaml.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_yaml}")

    extracted = episode_dir / "audio_extracted.wav"
    vocals = episode_dir / "audio_vocals.wav"
    enhanced_vocals = episode_dir / "audio_enhanced_vocals.wav"
    enhanced_mix = episode_dir / "audio_enhanced_mix.wav"
    report = episode_dir / "preprocessing_report.json"

    artifacts = {
        "extracted": extracted,
        "vocals": None,
        "enhanced_vocals": None,
        "enhanced_mix": None,
        "report": report,
    }

    try:
        ffmpeg_extract_wav(media_path, extracted)
    except AudioPreprocessingError as exc:
        exc.artifacts = dict(artifacts)
        exc.artifacts["extracted"] = extracted if extracted.exists() else None
        raise

    try:
        run_audio_preprocessing(extracted, vocals, enhanced_vocals, enhanced_mix, preset_yaml)
    except AudioPreprocessingError as exc:
        exc.artifacts = dict(artifacts)
        exc.artifacts["extracted"] = extracted if extracted.exists() else None
        raise

    artifacts["vocals"] = vocals if vocals.exists() else None
    artifacts["enhanced_vocals"] = enhanced_vocals if enhanced_vocals.exists() else None
    artifacts["enhanced_mix"] = enhanced_mix if enhanced_mix.exists() else None

    payload = {
        "media": str(media_path),
        "preset": str(preset_yaml),
        "outputs": {
            "extracted": str(extracted if extracted.exists() else ""),
            "vocals": str(vocals if vocals.exists() else ""),
            "enhanced_vocals": str(enhanced_vocals if enhanced_vocals.exists() else ""),
            "enhanced_mix": str(enhanced_mix if enhanced_mix.exists() else ""),
        },
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote preprocessing report to %s", report)
    LOGGER.info(
        "Enhanced vocals: %s", enhanced_vocals if enhanced_vocals.exists() else "not created"
    )
    LOGGER.info("Enhanced mix: %s", enhanced_mix if enhanced_mix.exists() else "not created")

    return artifacts
