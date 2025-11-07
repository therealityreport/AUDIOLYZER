#!/usr/bin/env python3
"""Verify that the Show-Scribe runtime dependencies are available."""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass

REQUIRED_BINARIES = ("ffmpeg", "ffprobe")
OPTIONAL_BINARIES = ("sox",)
REQUIRED_MODULES = (
    "yaml",
    "faster_whisper",
    "torch",
    "torchaudio",
    "pyannote.audio",
    "resemblyzer",
    "numpy",
    "pandas",
    "streamlit",
    "huggingface_hub",
    "requests",
)
OPTIONAL_MODULES = ("rich", "typer")


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""
    optional: bool = False

    def render(self) -> str:
        if self.ok:
            status = "OK "
        elif self.optional:
            status = "WARN"
        else:
            status = "ERR"
        return f"[{status}] {self.name}{' - ' + self.details if self.details else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate external binaries and Python packages required by Show-Scribe."
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependency checks and only validate required components.",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Report whether GPU acceleration is available via PyTorch.",
    )
    args = parser.parse_args()

    results: list[CheckResult] = []
    results.extend(check_python_version())
    results.extend(check_binaries(REQUIRED_BINARIES, required=True))
    if not args.skip_optional:
        results.extend(check_binaries(OPTIONAL_BINARIES, required=False))
    results.extend(check_modules(REQUIRED_MODULES, required=True))
    if not args.skip_optional:
        results.extend(check_modules(OPTIONAL_MODULES, required=False))

    if args.check_gpu:
        results.extend(check_gpu_support())

    failures = [result for result in results if not result.ok and not result.optional]
    for result in results:
        print(result.render())

    if failures:
        print(
            "\nSome required dependencies are missing. "
            "Install the packages above before continuing."
        )
        raise SystemExit(1)


def check_python_version(minimum: tuple[int, int] = (3, 11)) -> list[CheckResult]:
    version_info = sys.version_info
    current = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    expected = ".".join(str(part) for part in minimum)
    ok = (version_info.major, version_info.minor) >= minimum
    details = f"Detected Python {current}; requires >= {expected}"
    return [CheckResult(name="python", ok=ok, details=details)]


def check_binaries(names: Iterable[str], *, required: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in names:
        path = shutil.which(name)
        if path:
            details = f"found at {path}"
            results.append(CheckResult(name=name, ok=True, details=details))
        else:
            qualifier = "required" if required else "optional"
            results.append(
                CheckResult(
                    name=name,
                    ok=False,
                    optional=not required,
                    details=f"{qualifier} binary missing",
                )
            )
    return results


def check_modules(names: Iterable[str], *, required: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    for module_name in names:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            qualifier = "required" if required else "optional"
            results.append(
                CheckResult(
                    name=module_name,
                    ok=False,
                    optional=not required,
                    details=f"{qualifier} module missing ({exc.name})",
                )
            )
        else:
            results.append(CheckResult(name=module_name, ok=True))
    return results


def check_gpu_support() -> list[CheckResult]:
    try:
        import torch
    except ModuleNotFoundError:
        return [
            CheckResult(
                name="gpu",
                ok=False,
                optional=True,
                details="PyTorch not installed",
            )
        ]

    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None)
    apple_mps = bool(mps_available and torch.backends.mps.is_available())  # type: ignore[attr-defined]
    details = []
    details.append(f"CUDA={'yes' if cuda_available else 'no'}")
    details.append(f"MPS={'yes' if apple_mps else 'no'}")
    details.append(f"device={platform.platform()}")
    return [
        CheckResult(
            name="gpu",
            ok=cuda_available or apple_mps,
            optional=True,
            details=", ".join(details),
        )
    ]


if __name__ == "__main__":
    main()
