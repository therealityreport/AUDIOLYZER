#!/usr/bin/env python3
"""Model downloader for sherpa-onnx models."""

import argparse
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Model registry for sherpa-onnx
SHERPA_MODELS = {
    "paraformer-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-en-2023-10-24.tar.bz2",
        "files": ["model.onnx", "tokens.txt", "decoder.onnx"],
        "required": ["model.onnx", "tokens.txt"],
        "description": "Paraformer English ASR model",
    },
    "paraformer-zh": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2",
        "files": ["model.onnx", "tokens.txt"],
        "required": ["model.onnx", "tokens.txt"],
        "description": "Paraformer Chinese ASR model",
    },
    "zipformer-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        "files": ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        "required": ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        "description": "Zipformer English ASR model (transducer)",
    },
    "zipformer-zh": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-2023-10-24.tar.bz2",
        "files": ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        "required": ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        "description": "Zipformer Chinese ASR model (transducer)",
    },
}


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress reporting.

    Args:
        url: URL to download
        output_path: Output file path
    """

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\r[{bar}] {percent:.1f}%", end="", flush=True)

    print(f"Downloading {url}")
    urlretrieve(url, output_path, reporthook=_progress)
    print()  # New line after progress


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract tar.bz2 or zip archive.

    Args:
        archive_path: Path to archive file
        output_dir: Output directory

    Returns:
        Path to extracted directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
            # Get the root directory name
            extracted_dirs = [name for name in zf.namelist() if "/" in name]
            if extracted_dirs:
                root_dir = extracted_dirs[0].split("/")[0]
                return output_dir / root_dir

    elif archive_path.name.endswith(".tar.bz2"):
        with tarfile.open(archive_path, "r:bz2") as tf:
            tf.extractall(output_dir)
            # Get the root directory name
            members = tf.getmembers()
            if members:
                root_dir = members[0].name.split("/")[0]
                return output_dir / root_dir

    elif archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)
            members = tf.getmembers()
            if members:
                root_dir = members[0].name.split("/")[0]
                return output_dir / root_dir

    return output_dir


def validate_model_files(model_dir: Path, required_files: list) -> bool:
    """Validate that all required model files exist.

    Args:
        model_dir: Model directory
        required_files: List of required file names

    Returns:
        True if all files exist
    """
    missing = []
    for filename in required_files:
        if not (model_dir / filename).exists():
            missing.append(filename)

    if missing:
        logger.error(f"Missing required files: {', '.join(missing)}")
        return False

    return True


def download_sherpa_model(model_name: str, output_dir: Optional[Path] = None) -> Path:
    """Download and extract a sherpa-onnx model.

    Args:
        model_name: Model name (e.g., "paraformer-en", "zipformer-en")
        output_dir: Output directory (default: models/sherpa-onnx/)

    Returns:
        Path to extracted model directory
    """
    if model_name not in SHERPA_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}\n" f"Available models: {', '.join(SHERPA_MODELS.keys())}"
        )

    model_info = SHERPA_MODELS[model_name]

    # Set default output directory
    if output_dir is None:
        output_dir = Path("models/sherpa-onnx")
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download archive
    url = model_info["url"]
    archive_name = url.split("/")[-1]
    archive_path = output_dir / archive_name

    if not archive_path.exists():
        logger.info(f"Downloading {model_name}: {model_info['description']}")
        download_file(url, archive_path)
    else:
        logger.info(f"Archive already exists: {archive_path}")

    # Extract
    logger.info(f"Extracting {archive_path}")
    extracted_dir = extract_archive(archive_path, output_dir)

    # Validate
    logger.info(f"Validating model files in {extracted_dir}")
    if not validate_model_files(extracted_dir, model_info["required"]):
        raise RuntimeError(
            f"Model validation failed. Required files: {', '.join(model_info['required'])}\n"
            f"Please check {extracted_dir}"
        )

    logger.info(f"✓ Model ready: {extracted_dir}")

    # Cleanup archive
    try:
        archive_path.unlink()
        logger.info(f"Cleaned up archive: {archive_path}")
    except Exception as e:
        logger.warning(f"Failed to remove archive: {e}")

    return extracted_dir


def list_models() -> None:
    """List available models."""
    print("Available sherpa-onnx models:")
    print()
    for name, info in SHERPA_MODELS.items():
        print(f"  {name:20} - {info['description']}")
        print(f"  {'':20}   Required files: {', '.join(info['required'])}")
        print()


def check_model_exists(model_name: str, model_dir: Optional[Path] = None) -> bool:
    """Check if a model is already downloaded and valid.

    Args:
        model_name: Model name
        model_dir: Model directory to check

    Returns:
        True if model exists and is valid
    """
    if model_name not in SHERPA_MODELS:
        return False

    model_info = SHERPA_MODELS[model_name]

    if model_dir is None:
        # Try to find it in default location
        model_dir = Path("models/sherpa-onnx")
        # Look for subdirectories that might contain the model
        if model_dir.exists():
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and validate_model_files(subdir, model_info["required"]):
                    return True
        return False

    return validate_model_files(Path(model_dir), model_info["required"])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download sherpa-onnx models",
        epilog="Example: python download_models.py --sherpa paraformer-en",
    )

    parser.add_argument(
        "--sherpa",
        metavar="MODEL",
        help="Download sherpa-onnx model (paraformer-en, zipformer-en, etc.)",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: models/sherpa-onnx/)"
    )
    parser.add_argument("--check", metavar="MODEL", help="Check if model exists and is valid")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Handle commands
    if args.list:
        list_models()
        return

    if args.check:
        exists = check_model_exists(args.check, args.output_dir)
        if exists:
            print(f"✓ Model '{args.check}' is installed and valid")
            sys.exit(0)
        else:
            print(f"✗ Model '{args.check}' not found or invalid")
            print(f"\nDownload with: python {sys.argv[0]} --sherpa {args.check}")
            sys.exit(1)

    if args.sherpa:
        try:
            model_dir = download_sherpa_model(args.sherpa, args.output_dir)
            print(f"\n✓ Success! Model installed at: {model_dir}")
            print(f"\nUpdate your eval.yaml:")
            print(f"  asr:")
            print(f"    sherpa_onnx:")
            print(f"      enabled: true")
            print(f'      model_path: "{model_dir}"')
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
