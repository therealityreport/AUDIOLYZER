#!/usr/bin/env python3
"""Re-run diarization on BARBADOS episode with fixed config."""

from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from show_scribe.ui_ops import run_diarization

# BARBADOS episode directory
episode_dir = Path("/Volumes/HardDrive/AUDIO ANALYZER/data/shows/RHOSLC/episodes/BARBADOS")

print(f"Re-running diarization on {episode_dir.name}...")
print("Using updated config with overlap_threshold: null (pyannote default ~0.7)")
print()

result = run_diarization(episode_dir)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

sys.exit(result.returncode)
