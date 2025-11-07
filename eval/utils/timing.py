"""Timing utilities for measuring runtime and computing RTF."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import wave

logger = logging.getLogger(__name__)


class Timer:
    """Simple timer for measuring elapsed time."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@contextmanager
def time_operation(operation_name: str, log_level: int = logging.INFO):
    """Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed
        log_level: Logging level for the timing message

    Yields:
        Timer object

    Example:
        >>> with time_operation("ASR processing") as timer:
        ...     process_audio()
        >>> print(f"Took {timer.elapsed:.2f}s")
    """
    timer = Timer()
    timer.start()

    try:
        yield timer
    finally:
        elapsed = timer.stop()
        logger.log(log_level, f"{operation_name} completed in {elapsed:.2f}s")


def get_audio_duration(audio_path: str | Path) -> float:
    """Get duration of audio file in seconds.

    Args:
        audio_path: Path to WAV file

    Returns:
        Duration in seconds
    """
    audio_path = Path(audio_path)

    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)

    return duration


def compute_rtf(audio_duration: float, processing_time: float) -> float:
    """Compute Real-Time Factor (RTF).

    RTF = processing_time / audio_duration

    RTF < 1.0 means faster than real-time
    RTF = 1.0 means real-time
    RTF > 1.0 means slower than real-time

    Args:
        audio_duration: Duration of audio in seconds
        processing_time: Processing time in seconds

    Returns:
        RTF value
    """
    if audio_duration <= 0:
        raise ValueError("Audio duration must be positive")

    return processing_time / audio_duration


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s" or "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    secs = seconds % 60

    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m {secs:.0f}s"


class RuntimeLogger:
    """Logger for runtime statistics."""

    def __init__(self, log_file: str | Path):
        """Initialize runtime logger.

        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[Dict[str, Any]] = []

    def log_run(
        self,
        clip: str,
        tool: str,
        stage: str,
        duration: float,
        audio_duration: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log a single run.

        Args:
            clip: Clip name
            tool: Tool name
            stage: Processing stage (e.g., "asr", "diarization")
            duration: Processing duration in seconds
            audio_duration: Audio duration in seconds (for RTF calculation)
            **kwargs: Additional metadata
        """
        entry = {
            "clip": clip,
            "tool": tool,
            "stage": stage,
            "duration_s": round(duration, 3),
            "timestamp": time.time(),
        }

        if audio_duration is not None:
            entry["audio_duration_s"] = round(audio_duration, 3)
            entry["rtf"] = round(compute_rtf(audio_duration, duration), 4)

        entry.update(kwargs)
        self.entries.append(entry)

    def save(self) -> None:
        """Save runtime log to file."""
        import json

        with open(self.log_file, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Saved {len(self.entries)} runtime entries to {self.log_file}")

    def get_stats(
        self, tool: Optional[str] = None, stage: Optional[str] = None
    ) -> Dict[str, float]:
        """Get statistics for logged runs.

        Args:
            tool: Filter by tool name
            stage: Filter by stage

        Returns:
            Dictionary with statistics
        """
        filtered = self.entries

        if tool is not None:
            filtered = [e for e in filtered if e.get("tool") == tool]

        if stage is not None:
            filtered = [e for e in filtered if e.get("stage") == stage]

        if not filtered:
            return {}

        durations = [e["duration_s"] for e in filtered]
        rtfs = [e["rtf"] for e in filtered if "rtf" in e]

        stats = {
            "count": len(filtered),
            "total_duration": sum(durations),
            "mean_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
        }

        if rtfs:
            stats["mean_rtf"] = sum(rtfs) / len(rtfs)
            stats["min_rtf"] = min(rtfs)
            stats["max_rtf"] = max(rtfs)

        return stats
