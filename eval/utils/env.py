"""Environment manifest logging for reproducibility."""

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_python_version() -> str:
    """Get Python version string.

    Returns:
        Python version (e.g., "3.11.7")
    """
    return platform.python_version()


def get_platform_info() -> Dict[str, str]:
    """Get platform information.

    Returns:
        Dictionary with platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def get_package_versions() -> Dict[str, str]:
    """Get installed package versions.

    Returns:
        Dictionary mapping package names to versions
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = json.loads(result.stdout)
        return {pkg["name"]: pkg["version"] for pkg in packages}

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to get package versions: {e}")
        return {}


def get_git_info() -> Dict[str, str]:
    """Get git repository information.

    Returns:
        Dictionary with git info (commit hash, branch, etc.)
    """
    git_info = {}

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        git_info["commit"] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        git_info["branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        git_info["dirty"] = bool(result.stdout.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repository or git not available
        pass

    return git_info


def create_env_manifest() -> Dict[str, Any]:
    """Create environment manifest for reproducibility.

    Returns:
        Dictionary with complete environment information
    """
    manifest = {
        "python_version": get_python_version(),
        "platform": get_platform_info(),
        "packages": get_package_versions(),
        "git": get_git_info(),
    }

    return manifest


def save_env_manifest(output_path: str | Path) -> None:
    """Save environment manifest to JSON file.

    Args:
        output_path: Path to save manifest
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = create_env_manifest()

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved environment manifest to {output_path}")


def print_env_summary() -> None:
    """Print summary of environment information."""
    manifest = create_env_manifest()

    print("Environment Information:")
    print(f"  Python: {manifest['python_version']}")
    print(f"  System: {manifest['platform']['system']} {manifest['platform']['release']}")
    print(f"  Machine: {manifest['platform']['machine']}")

    if manifest["git"]:
        print(
            f"  Git: {manifest['git'].get('branch', 'N/A')} @ {manifest['git'].get('commit', 'N/A')[:8]}"
        )
        if manifest["git"].get("dirty"):
            print("  ⚠️  Uncommitted changes present")

    # Print key package versions
    packages = manifest["packages"]
    key_packages = [
        "torch",
        "numpy",
        "faster-whisper",
        "sherpa-onnx",
        "pyannote.audio",
        "speechbrain",
    ]

    print("  Key packages:")
    for pkg in key_packages:
        version = packages.get(pkg, "not installed")
        print(f"    {pkg}: {version}")
