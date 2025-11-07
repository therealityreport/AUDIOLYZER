#!/usr/bin/env python3
"""Vocal separation module for audio preprocessing."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Use Demucs instead of audio-separator for better macOS compatibility
import subprocess
import shutil


def load_config(preset_path: Path) -> Dict[str, Any]:
    """Load configuration from preset YAML file."""
    import yaml

    with open(preset_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("audio_preprocessing", {}).get("vocal_separation", {})


def main():
    """Main entry point for vocal separation."""
    parser = argparse.ArgumentParser(description="Separate vocals from background music/noise")
    parser.add_argument("--preset", required=True, help="Path to preset YAML file")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output vocals file path")
    parser.add_argument("--stderr-log", help="Path to write stderr output for debugging")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    preset_path = Path(args.preset)
    stderr_log_path = Path(args.stderr_log) if args.stderr_log else None

    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not preset_path.exists():
        print(f"Error: Preset file does not exist: {preset_path}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_config(preset_path)
    model = config.get("model", "htdemucs")
    segment_seconds = config.get("segment_seconds", None)
    overlap = config.get("overlap", None)
    shifts = config.get("shifts", 1)  # TTA shifts, 0=disabled, 1=no extra passes
    device = config.get("device", "auto")
    dtype = config.get("dtype", None)  # fp16/float32, ignored by demucs (uses fp32 internally)
    timeout_seconds = config.get("timeout_seconds", 1200)

    # Try to use mdx_extra, fall back to mdx if it doesn't exist
    if model == "mdx_extra":
        print(
            "Note: mdx_extra may not exist in this demucs version, will fall back to mdx if needed"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if importlib.util.find_spec("demucs.separate") is None:
            print(
                "Error: Demucs is not installed. Reinstall project dependencies (e.g. `pip install -e .`) "
                "or install it directly with `pip install demucs`.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Use Demucs for vocal separation
        print(f"Separating vocals using Demucs model: {model}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Create temporary output directory
        temp_output_dir = output_path.parent / "temp_separation"
        temp_output_dir.mkdir(exist_ok=True)

        # Build Demucs command with performance parameters
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--name",
            model,
            "--out",
            str(temp_output_dir),
            "--two-stems",
            "vocals",  # Only separate vocals and instrumental
        ]

        # Add performance parameters if specified
        if segment_seconds is not None:
            cmd.extend(["--segment", str(segment_seconds)])
        if overlap is not None:
            cmd.extend(["--overlap", str(overlap)])
        if shifts is not None and shifts >= 0:
            cmd.extend(["--shifts", str(shifts)])
        if device and device != "auto":
            cmd.extend(["--device", device])

        cmd.append(str(input_path))

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)

        if result.returncode != 0:
            error_msg = f"Demucs failed with exit code {result.returncode}\n"
            error_msg += f"stderr: {result.stderr}\n"
            error_msg += f"stdout: {result.stdout}\n"
            print(error_msg, file=sys.stderr)

            # Write stderr to log file if specified
            if stderr_log_path:
                stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(stderr_log_path, "w", encoding="utf-8") as f:
                    f.write(error_msg)

            sys.exit(1)

        # Find the vocals file in the output
        # Demucs creates: temp_separation/model_name/track_name/vocals.wav
        vocals_file = None
        for model_dir in temp_output_dir.iterdir():
            if model_dir.is_dir():
                for track_dir in model_dir.iterdir():
                    if track_dir.is_dir():
                        vocals_candidate = track_dir / "vocals.wav"
                        if vocals_candidate.exists():
                            vocals_file = vocals_candidate
                            break
                if vocals_file:
                    break

        if vocals_file and vocals_file.exists():
            # Copy the vocals file to the expected output location
            shutil.copy2(vocals_file, output_path)
            print(f"✓ Vocals separated successfully: {output_path}")

            # Clean up temporary directory
            shutil.rmtree(temp_output_dir)
        else:
            print(f"Error: Vocals file not found in Demucs output", file=sys.stderr)
            print(f"Searched in: {temp_output_dir}", file=sys.stderr)
            sys.exit(1)

    except subprocess.TimeoutExpired:
        error_msg = f"Error: Demucs separation timed out after {timeout_seconds}s"
        print(error_msg, file=sys.stderr)
        if stderr_log_path:
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stderr_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error during vocal separation: {e}"
        print(error_msg, file=sys.stderr)
        if stderr_log_path:
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stderr_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
