#!/usr/bin/env python3
"""Download the machine learning models required by Show-Scribe."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from show_scribe.config.load import ConfigError, load_config

LOGGER = logging.getLogger("download_models")

DEFAULT_MODELS = ("whisper", "pyannote", "resemblyzer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Whisper, Pyannote, and Resemblyzer models used by Show-Scribe."
    )
    parser.add_argument(
        "--env",
        default=os.environ.get("SHOW_SCRIBE_ENVIRONMENT", "dev"),
        help="Configuration environment to read download locations from (default: %(default)s).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing environment YAML files "
            "(defaults to the repository configs/)."
        ),
    )
    parser.add_argument(
        "--models",
        choices=(*DEFAULT_MODELS, "all"),
        action="append",
        help="Subset of models to download. Repeat flag for multiple entries. Defaults to all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if they already exist locally.",
    )
    parser.add_argument(
        "--pyannote-token",
        default=os.environ.get("SHOW_SCRIBE_PYANNOTE_TOKEN"),
        help=(
            "Optional Hugging Face token for Pyannote models. "
            "Falls back to configured auth token environment variable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    try:
        config = load_config(args.env, config_dir=args.config_dir)
    except (FileNotFoundError, ConfigError) as exc:
        LOGGER.error("Unable to load configuration: %s", exc)
        raise SystemExit(1) from exc

    models_dir = Path(config["paths"]["models_dir"]).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    requested = resolve_requested_models(args.models)
    LOGGER.info("Preparing to download models: %s", ", ".join(requested))

    failures = False
    if "whisper" in requested:
        try:
            download_whisper_model(config, force=args.force)
        except Exception as exc:  # pragma: no cover - defensive logging
            failures = True
            LOGGER.error("Whisper download failed: %s", exc)
        else:
            LOGGER.info("✓ Whisper model ready.")

    if "pyannote" in requested:
        token = resolve_pyannote_token(config, explicit_token=args.pyannote_token)
        try:
            download_pyannote_model(config, token=token, force=args.force)
        except Exception as exc:  # pragma: no cover - defensive logging
            failures = True
            LOGGER.error("Pyannote download failed: %s", exc)
        else:
            LOGGER.info("✓ Pyannote model ready.")

    if "resemblyzer" in requested:
        try:
            download_resemblyzer_model(config, force=args.force)
        except Exception as exc:  # pragma: no cover - defensive logging
            failures = True
            LOGGER.error("Resemblyzer download failed: %s", exc)
        else:
            LOGGER.info("✓ Resemblyzer resources ready.")

    raise SystemExit(1 if failures else 0)


def resolve_requested_models(models: Iterable[str] | None) -> list[str]:
    """Resolve the set of models that should be downloaded."""
    if not models:
        return list(DEFAULT_MODELS)

    resolved: set[str] = set()
    for entry in models:
        if entry == "all":
            resolved.update(DEFAULT_MODELS)
        else:
            resolved.add(entry)
    return sorted(resolved)


def download_whisper_model(config: dict[str, object], *, force: bool) -> None:
    """Download the Whisper model configured for the environment."""
    whisper_cfg = config["providers"]["whisper"]
    assert isinstance(whisper_cfg, dict)
    model_name = str(whisper_cfg["model"])
    download_root = Path(str(whisper_cfg["download_root"])).expanduser()
    download_root.mkdir(parents=True, exist_ok=True)

    target_dir = download_root / model_name
    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Run `pip install faster-whisper`."
        ) from exc

    try:
        WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(download_root),
        )
    except Exception as exc:  # pragma: no cover - download failures
        raise RuntimeError(f"Failed to download Whisper model '{model_name}': {exc}") from exc


def download_pyannote_model(
    config: dict[str, object],
    *,
    token: str | None,
    force: bool,
) -> None:
    """Download the Pyannote diarization model via Hugging Face Hub."""
    try:
        from huggingface_hub import HfHubHTTPError, snapshot_download
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. " "Run `pip install huggingface_hub`."
        ) from exc

    provider_cfg = config["providers"]["pyannote"]
    assert isinstance(provider_cfg, dict)
    model_id = str(provider_cfg["model"])

    base_dir = Path(config["paths"]["models_dir"]).expanduser()
    local_dir = base_dir / "pyannote" / sanitize_model_id(model_id)
    if force and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": not force,
        "force_download": force,
    }
    if token:
        kwargs["token"] = token
    else:
        LOGGER.warning(
            "No Hugging Face token provided for Pyannote downloads. "
            "Private models will fail; set SHOW_SCRIBE_PYANNOTE_TOKEN."
        )

    try:
        snapshot_download(**kwargs)
    except HfHubHTTPError as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"Hugging Face download failed for '{model_id}': {exc}") from exc


def download_resemblyzer_model(config: dict[str, object], *, force: bool) -> None:
    """Download the Resemblyzer VAD resources."""
    try:
        from huggingface_hub import HfHubHTTPError, snapshot_download
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. " "Run `pip install huggingface_hub`."
        ) from exc

    provider_cfg = config["providers"]["resemblyzer"]
    assert isinstance(provider_cfg, dict)
    model_id = str(provider_cfg["model"])

    base_dir = Path(config["paths"]["models_dir"]).expanduser()
    local_dir = base_dir / "resemblyzer" / sanitize_model_id(model_id)
    if force and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": not force,
        "force_download": force,
    }

    try:
        snapshot_download(**kwargs)
    except HfHubHTTPError as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"Hugging Face download failed for '{model_id}': {exc}") from exc


def resolve_pyannote_token(config: dict[str, object], *, explicit_token: str | None) -> str | None:
    """Determine which token to use for Pyannote downloads."""
    if explicit_token:
        return explicit_token

    provider_cfg = config["providers"]["pyannote"]
    assert isinstance(provider_cfg, dict)
    env_var = provider_cfg.get("auth_token_env")
    if not env_var:
        return None
    return os.environ.get(str(env_var))


def sanitize_model_id(model_id: str) -> str:
    """Convert a Hugging Face model identifier into a filesystem-safe name."""
    return model_id.replace("/", "__")


if __name__ == "__main__":
    main()
