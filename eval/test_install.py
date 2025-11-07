#!/usr/bin/env python3
"""Test script to verify installation of eval harness dependencies."""

import shutil
import subprocess
import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    tests = {
        "yaml (PyYAML)": lambda: __import__("yaml"),
        "numpy": lambda: __import__("numpy"),
        "torch": lambda: __import__("torch"),
        "faster_whisper": lambda: __import__("faster_whisper"),
        "pyannote.audio": lambda: __import__("pyannote.audio"),
    }

    optional_tests = {
        "sherpa_onnx": lambda: __import__("sherpa_onnx"),
        "speechbrain": lambda: __import__("speechbrain"),
        "dscore": lambda: __import__("dscore"),
    }

    failed = []

    # Test required imports
    for name, test_func in tests.items():
        try:
            test_func()
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)

    # Test optional imports
    print("\nOptional dependencies:")
    for name, test_func in optional_tests.items():
        try:
            test_func()
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} (optional)")

    return len(failed) == 0


def test_ffmpeg():
    """Test that ffmpeg is available."""
    print("\nTesting ffmpeg...")
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("  ✗ ffmpeg not found in PATH")
        print("    Install with: brew install ffmpeg")
        return False

    try:
        result = subprocess.run(  # noqa: S603
            [ffmpeg_path, "-version"],
            capture_output=True,
            check=True,
            text=True,
        )
        version_line = result.stdout.splitlines()[0]
        print(f"  ✓ {version_line}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ✗ ffmpeg not found: {e}")
        print("    Install with: brew install ffmpeg")
        return False


def test_huggingface_auth():
    """Test Hugging Face authentication."""
    print("\nTesting Hugging Face authentication...")
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            print("  ✓ Hugging Face token found")
            return True
        print("  ⚠ No Hugging Face token found")
        print("    Run: huggingface-cli login")
        print("    Then accept model terms at:")
        print("    - https://huggingface.co/pyannote/speaker-diarization@2.1")
        print("    - https://huggingface.co/pyannote/segmentation-3.0")
        return False
    except ImportError:
        print("  ⚠ huggingface_hub not installed")
        print("    Install with: pip install huggingface_hub[cli]")
        return False


def test_module_structure():
    """Test that eval module structure is correct."""
    print("\nTesting eval module structure...")

    modules = [
        "eval.utils",
        "eval.asr",
        "eval.diar",
        "eval.align",
        "eval.score",
    ]

    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False

    return all_ok


def test_config():
    """Test that config file exists and is valid."""
    print("\nTesting configuration...")

    from pathlib import Path

    config_path = Path("eval/eval.yaml")
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Config file valid: {config_path}")

        # Check required keys
        required_keys = ["paths", "audio", "asr", "diarization"]
        missing = [k for k in required_keys if k not in config]

        if missing:
            print(f"  ⚠ Missing config keys: {missing}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Config file invalid: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Eval Harness Installation Test")
    print("=" * 60)

    results = []

    results.append(("Dependencies", test_imports()))
    results.append(("ffmpeg", test_ffmpeg()))
    results.append(("Hugging Face", test_huggingface_auth()))
    results.append(("Module structure", test_module_structure()))
    results.append(("Configuration", test_config()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run eval harness.")
        print("\nNext steps:")
        print("  1. Add audio clips to data/clips/")
        print("  2. Run: python eval/runner.py --config eval/eval.yaml")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nFor help, see:")
        print("  - QUICKSTART.md")
        print("  - README.md")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
