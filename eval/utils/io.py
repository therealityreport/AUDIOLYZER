"""I/O utilities for eval harness."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to eval.yaml

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, output_path: str | Path, pretty: bool = True) -> None:
    """Save data as JSON.

    Args:
        data: Data to serialize
        output_path: Output file path
        pretty: If True, use indentation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)

    logger.debug(f"Saved JSON to {output_path}")


def load_json(input_path: str | Path) -> Any:
    """Load JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], output_path: str | Path) -> None:
    """Save records as JSONL (one JSON object per line).

    Args:
        records: List of dictionaries
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug(f"Saved {len(records)} records to {output_path}")


def load_jsonl(input_path: str | Path) -> List[Dict]:
    """Load JSONL file.

    Args:
        input_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_rttm(segments: List[tuple], output_path: str | Path, file_id: str) -> None:
    """Save diarization segments as RTTM.

    RTTM format:
    SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    Args:
        segments: List of (start, end, speaker_id) tuples
        output_path: Output RTTM file path
        file_id: File identifier for RTTM
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for start, end, speaker_id in segments:
            duration = end - start
            f.write(
                f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
            )

    logger.debug(f"Saved RTTM with {len(segments)} segments to {output_path}")


def load_rttm(input_path: str | Path) -> List[tuple]:
    """Load RTTM file.

    Args:
        input_path: Path to RTTM file

    Returns:
        List of (start, end, speaker_id) tuples
    """
    segments = []
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
                speaker_id = parts[7]
                segments.append((start, end, speaker_id))

    return segments


def get_audio_files(directory: str | Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Get all audio/video files in directory.

    Args:
        directory: Directory to search
        extensions: List of file extensions (default: common audio/video formats)

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".mp4", ".mov", ".m4a", ".flac", ".ogg"]

    directory = Path(directory)
    files = []

    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(files)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
