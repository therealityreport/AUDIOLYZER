#!/usr/bin/env python3
"""
Install audio preprocessing dependencies for Show-Scribe.

This script installs:
- audio-separator (for vocal separation)
- resemble-enhance (for audio enhancement)
- clearvoice (for ClearerVoice Studio integration)
"""

import subprocess
import sys


def run_command(cmd: list[str]) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(e.stderr)
        return False


def main():
    """Install preprocessing dependencies."""
    print("=" * 60)
    print("Show-Scribe Audio Preprocessing Setup")
    print("=" * 60)
    print()

    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 11):
        print("❌ Python 3.11+ required")
        sys.exit(1)
    print("✓ Python version OK")
    print()

    # Install audio-separator
    print("Installing audio-separator (vocal separation)...")
    print("-" * 60)
    if not run_command([sys.executable, "-m", "pip", "install", "audio-separator"]):
        print("❌ Failed to install audio-separator")
        sys.exit(1)
    print("✓ audio-separator installed")
    print()

    # Install resemble-enhance
    print("Installing resemble-enhance (audio enhancement)...")
    print("-" * 60)
    if not run_command([sys.executable, "-m", "pip", "install", "resemble-enhance"]):
        print("❌ Failed to install resemble-enhance")
        sys.exit(1)
    print("✓ resemble-enhance installed")
    print()

    # Install clearvoice
    print("Installing clearvoice (ClearerVoice Studio)...")
    print("-" * 60)
    if not run_command([sys.executable, "-m", "pip", "install", "clearvoice"]):
        print("❌ Failed to install clearvoice")
        sys.exit(1)
    print("✓ clearvoice installed")
    print()

    # Install torch if not already installed (required by resemble-enhance)
    print("Checking torch installation...")
    try:
        import torch

        print(f"✓ torch {torch.__version__} already installed")
    except ImportError:
        print("Installing torch...")
        if not run_command([sys.executable, "-m", "pip", "install", "torch"]):
            print("❌ Failed to install torch")
            sys.exit(1)
        print("✓ torch installed")
    print()

    # Install torchaudio if not already installed
    print("Checking torchaudio installation...")
    try:
        import torchaudio

        print(f"✓ torchaudio {torchaudio.__version__} already installed")
    except ImportError:
        print("Installing torchaudio...")
        if not run_command([sys.executable, "-m", "pip", "install", "torchaudio"]):
            print("❌ Failed to install torchaudio")
            sys.exit(1)
        print("✓ torchaudio installed")
    print()

    # Verify installations
    print("Verifying installations...")
    print("-" * 60)

    try:
        from audio_separator.separator import Separator

        print("✓ audio-separator can be imported")
    except ImportError as e:
        print(f"❌ Failed to import audio-separator: {e}")
        sys.exit(1)

    try:
        from resemble_enhance.enhancer.inference import enhance

        print("✓ resemble-enhance can be imported")
    except ImportError as e:
        print(f"❌ Failed to import resemble-enhance: {e}")
        sys.exit(1)

    try:
        from clearvoice import ClearVoice  # noqa: F401

        print("✓ clearvoice can be imported")
    except ImportError as e:
        print(f"❌ Failed to import clearvoice: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✓ All preprocessing dependencies installed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Update your config to enable preprocessing:")
    print("   cp configs/reality_tv.yaml configs/my_show.yaml")
    print()
    print("2. Process an episode with the new config:")
    print("   show-scribe process episode.mp4 --config configs/my_show.yaml")
    print()
    print("3. Compare transcription quality before/after preprocessing")
    print()


if __name__ == "__main__":
    main()
