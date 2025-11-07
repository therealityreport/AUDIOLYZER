"""FFmpeg command wrappers and helpers."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import cast

__all__ = [
    "FFmpeg",
    "FFmpegError",
    "FFmpegProgress",
    "LoudnessSettings",
]


Callback = Callable[["FFmpegProgress"], None]


class FFmpegError(RuntimeError):
    """Raised when FFmpeg exits with a non-zero status."""


@dataclass(slots=True)
class FFmpegProgress:
    """Represents a parsed progress update from FFmpeg."""

    frame: int | None = None
    fps: float | None = None
    bitrate_kbps: float | None = None
    speed: float | None = None
    out_time: float | None = None
    total_size_kb: float | None = None
    status: str | None = None


@dataclass(slots=True)
class LoudnessSettings:
    """Settings for FFmpeg loudness normalization."""

    target_lufs: float = -20.0
    loudness_range: float = 7.0
    true_peak: float = -1.0
    dual_pass: bool = False

    def to_filter(self) -> str:
        """Return the ffmpeg filter string."""
        values = [
            f"I={self.target_lufs}",
            f"LRA={self.loudness_range}",
            f"TP={self.true_peak}",
        ]
        return f"loudnorm={':'.join(values)}"


class FFmpeg:
    """Lightweight wrapper around FFmpeg and FFprobe commands."""

    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        loudness: LoudnessSettings | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.loudness = loudness or LoudnessSettings()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def probe(self, media_path: str | Path) -> dict[str, object]:
        """Return metadata for the provided media file using ffprobe."""
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ]
        result = subprocess.run(  # noqa: S603 - command constructed from trusted input
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FFmpegError(
                f"ffprobe failed with code {result.returncode}: {result.stderr.strip()}"
            )
        payload = json.loads(result.stdout or "{}")
        if not isinstance(payload, dict):
            raise FFmpegError("ffprobe did not return a JSON object.")
        return cast(dict[str, object], payload)

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        audio_codec: str = "pcm_s16le",
        progress: Callback | None = None,
        extra_filters: Iterable[str] | None = None,
    ) -> None:
        """Extract and normalize audio from the input media."""
        filters = [self.loudness.to_filter()]
        if extra_filters:
            filters.extend(extra_filters)

        command = [
            "-i",
            str(input_path),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-vn",
            "-c:a",
            audio_codec,
            "-af",
            ",".join(filters),
            str(output_path),
        ]

        self.run(command, progress_callback=progress)

    def run(
        self,
        args: Sequence[str],
        *,
        progress_callback: Callback | None = None,
        timeout: float | None = None,
    ) -> None:
        """Invoke FFmpeg with the provided arguments."""
        base = [self.ffmpeg_path, "-hide_banner", "-y"]
        if progress_callback:
            base.extend(["-progress", "pipe:1", "-nostats"])
        command = [*base, *args]

        if progress_callback:
            self._run_with_progress(command, progress_callback, timeout=timeout)
        else:
            result = (
                subprocess.run(  # noqa: S603 - command is constructed from trusted configuration
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            )
            if result.returncode != 0:
                raise FFmpegError(result.stderr.strip())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_with_progress(
        self,
        command: Sequence[str],
        callback: Callback,
        *,
        timeout: float | None,
    ) -> None:
        start_time = time.monotonic()
        process = (
            subprocess.Popen(  # noqa: S603 - command is constructed from trusted configuration
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        )
        try:
            assert process.stdout is not None
            state = FFmpegProgress()
            while True:
                if timeout is not None and (time.monotonic() - start_time) > timeout:
                    process.kill()
                    raise FFmpegError("FFmpeg process timed out.")

                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue

                parsed = self._parse_progress_line(line)
                if not parsed:
                    continue

                key, value = parsed
                self._update_progress(state, key, value)
                if key == "progress":
                    callback(replace(state))
        finally:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise FFmpegError((stderr or stdout or "").strip())

    @staticmethod
    def _parse_progress_line(line: str) -> tuple[str, str] | None:
        line = line.strip()
        if not line or "=" not in line:
            return None
        key, _, value = line.partition("=")
        return key.strip(), value.strip()

    @staticmethod
    def _update_progress(progress: FFmpegProgress, key: str, value: str) -> None:
        if key == "frame":
            progress.frame = int(value)
        elif key == "fps":
            progress.fps = _parse_numeric(value)
        elif key == "bitrate":
            progress.bitrate_kbps = _parse_bitrate(value)
        elif key == "total_size":
            numeric = _parse_numeric(value)
            progress.total_size_kb = numeric / 1024.0 if numeric is not None else None
        elif key == "speed":
            progress.speed = _parse_speed(value)
        elif key == "out_time":
            progress.out_time = _parse_time(value)
        elif key == "out_time_ms":
            progress.out_time = float(value) / 1000.0
        elif key == "progress":
            progress.status = value


def _parse_bitrate(value: str) -> float | None:
    if value.endswith("kbits/s"):
        try:
            return float(value[:-7])
        except ValueError:
            return None
    return _parse_numeric(value)


def _parse_speed(value: str) -> float | None:
    if value.endswith("x"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    return _parse_numeric(value)


def _parse_numeric(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_time(value: str) -> float | None:
    try:
        hours, minutes, seconds = value.split(":")
        delta = timedelta(
            hours=int(hours),
            minutes=int(minutes),
            seconds=float(seconds),
        )
        return delta.total_seconds()
    except (ValueError, TypeError):
        return None
