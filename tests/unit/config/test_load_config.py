"""Tests for the configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

import show_scribe.config.load as config_load
from show_scribe.config.load import ConfigError, load_config


def test_load_config_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables prefixed with SHOW_SCRIBE_ override YAML values."""
    data_root = tmp_path / "custom"
    monkeypatch.setenv("SHOW_SCRIBE_LOGGING__LEVEL", "ERROR")
    monkeypatch.setenv("SHOW_SCRIBE_PATHS__DATA_ROOT", str(data_root))

    config = load_config("dev")

    assert config["logging"]["level"] == "ERROR"
    assert config["paths"]["data_root"] == str(data_root)


def test_load_config_validation_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid files raise ConfigError when validation fails."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    invalid_config = config_dir / "broken.yaml"
    invalid_config.write_text(
        "environment: dev\n" "paths:\n" "  data_root: ./data\n",  # missing many required fields
        encoding="utf-8",
    )

    @dataclass
    class DummyValidationError(Exception):
        path: list[str]
        message: str

    class DummyValidator:
        def __init__(self, _schema: dict[str, object]) -> None:
            pass

        def iter_errors(self, config: dict[str, object]):
            if "version" not in config:
                yield DummyValidationError(["version"], "'version' is a required property")

    # Ensure no local environment overrides interfere with validation outcome.
    for key in list(os.environ):
        if key.startswith("SHOW_SCRIBE_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(config_load, "Draft7Validator", DummyValidator, raising=False)
    monkeypatch.setattr(config_load, "ValidationError", DummyValidationError, raising=False)

    with pytest.raises(ConfigError):
        load_config("broken", config_dir=config_dir)
