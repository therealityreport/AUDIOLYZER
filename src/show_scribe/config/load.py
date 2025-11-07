"""Configuration loading helpers for Show-Scribe."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable, Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml

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

__all__ = ["ConfigError", "load_config"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
SCHEMA_PATH = Path(__file__).with_name("schema.json")
ENV_PREFIX = "SHOW_SCRIBE_"
ENV_SEPARATOR = "__"


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


def load_config(
    env: str = "dev",
    *,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Load the configuration for the requested environment.

    The lookup order is:
        1. Base YAML configuration in ``configs/{env}.yaml`` (or provided directory).
        2. Programmatic overrides supplied via the ``overrides`` mapping.
        3. Environment variables prefixed with ``SHOW_SCRIBE_`` using ``__`` as a nesting delimiter.

    Validation against ``schema.json`` is performed when the optional ``jsonschema`` dependency
    is available. Missing dependency results in a runtime warning instead of a hard failure.
    """

    base_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    config_path = _resolve_config_path(base_dir, env)
    config = _load_yaml_config(config_path)

    if overrides:
        config = _deep_merge(config, overrides)

    config = _apply_env_overrides(config)

    if validate:
        _validate_config(config)

    return config


def _resolve_config_path(base_dir: Path, env: str) -> Path:
    """Resolve the configuration file path for the requested environment."""
    for suffix in (".yaml", ".yml"):
        candidate = base_dir / f"{env}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Configuration file not found for environment '{env}' in {base_dir}.")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping in {path}.")
    return deepcopy(data)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    result: dict[str, Any] = deepcopy(dict(base))
    for key, override_value in overrides.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(override_value, Mapping)
        ):
            result[key] = _deep_merge(result[key], override_value)
        else:
            result[key] = deepcopy(override_value)
    return result


def _apply_env_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply overrides sourced from SHOW_SCRIBE_* environment variables."""
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue

        path_tokens = _parse_env_key(raw_key)
        if not path_tokens:
            continue

        value = _coerce_env_value(raw_value)
        _set_nested_value(overrides, path_tokens, value)

    if overrides:
        return _deep_merge(config, overrides)
    return deepcopy(dict(config))


def _parse_env_key(env_key: str) -> list[str]:
    """Turn an environment key into lowercase config tokens."""
    raw_path = env_key[len(ENV_PREFIX) :]
    if not raw_path:
        return []
    return [
        token.strip().lower().replace("-", "_")
        for token in raw_path.split(ENV_SEPARATOR)
        if token.strip()
    ]


def _set_nested_value(target: MutableMapping[str, Any], keys: Iterable[str], value: Any) -> None:
    """Set a value deeply in a nested mapping, creating dictionaries as needed."""
    keys = list(keys)
    if not keys:
        return

    current: MutableMapping[str, Any] = target
    for key in keys[:-1]:
        existing = current.get(key)
        if not isinstance(existing, MutableMapping):
            existing = {}
            current[key] = existing
        current = existing
    current[keys[-1]] = value


def _coerce_env_value(raw_value: str) -> Any:
    """Best-effort conversion from string to Python types using YAML parsing."""
    if raw_value == "":
        return ""
    try:
        parsed = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value
    return parsed


def _load_schema() -> dict[str, Any]:
    """Load the JSON schema definition for configuration validation."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Configuration schema must be a JSON object.")
    return cast(dict[str, Any], data)


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate the configuration against the JSON schema when possible."""
    if Draft7Validator is None or ValidationError is None:
        warnings.warn(
            "jsonschema is not installed; skipping configuration validation.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    raise ConfigError(f"Configuration validation failed:\n{error_message}") from errors[0]
