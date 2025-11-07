"""Helpers for loading per-show configuration JSON files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

_Draft7Validator: type[Any] | None
_ValidationError: type[Exception] | None

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft7Validator as _Draft7Validator
    from jsonschema.exceptions import ValidationError as _ValidationError
except ImportError:  # pragma: no cover - optional dependency
    _Draft7Validator = None
    _ValidationError = None

Draft7Validator: type[Any] | None = _Draft7Validator
ValidationError: type[Exception] | None = _ValidationError

__all__ = ["ShowConfigError", "load_show_config"]

SCHEMA_PATH = Path(__file__).with_name("show_config_schema.json")


class ShowConfigError(RuntimeError):
    """Raised when show configuration loading or validation fails."""


def load_show_config(config_path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load and optionally validate a per-show configuration."""
    path = Path(config_path)
    if not path.exists():
        raise ShowConfigError(f"Show configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ShowConfigError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ShowConfigError(f"Show configuration root must be an object: {path}")

    if validate:
        _validate_show_config(data)

    return data


def _validate_show_config(config: Mapping[str, Any]) -> None:
    """Validate the show configuration when jsonschema is available."""
    if Draft7Validator is None or ValidationError is None:
        return

    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.path))
    if not errors:
        return

    formatted = []
    for error in errors:
        path = ".".join(str(piece) for piece in error.path) or "<root>"
        formatted.append(f"- {path}: {error.message}")

    error_message = "\n".join(formatted)
    raise ShowConfigError(f"Show configuration validation failed:\n{error_message}") from errors[0]


def _load_schema() -> dict[str, Any]:
    """Load the JSON schema definition for show configuration files."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ShowConfigError("Show configuration schema must be a JSON object.")
    return cast(dict[str, Any], data)
