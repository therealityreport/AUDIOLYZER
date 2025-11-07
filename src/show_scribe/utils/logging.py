"""Project-wide logging configuration helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from logging import Handler, Logger
from pathlib import Path
from typing import Any

__all__ = ["configure_logging", "get_logger", "set_log_level"]

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(settings: Mapping[str, Any] | None = None, *, force: bool = True) -> None:
    """Configure the root logger according to the provided settings mapping.

    The mapping is expected to mirror the structure of the ``logging`` section inside the
    project configuration:

    .. code-block:: yaml

        logging:
          level: INFO
          json: false
          rich_tracebacks: true
          file:
            enabled: true
            path: /tmp/show-scribe.log
            rotation_mb: 20
            retention_days: 7

    Only a subset is honoured here (level and file logging). The remaining keys are accepted
    to keep the interface forwards compatible but are otherwise ignored.
    """

    level = _coerce_level(settings.get("level") if settings else None)

    logging.basicConfig(level=level, format=DEFAULT_FORMAT, force=force)

    file_settings = (settings or {}).get("file")
    handler: Handler | None = None
    if isinstance(file_settings, Mapping) and file_settings.get("enabled"):
        path_value = file_settings.get("path")
        if path_value:
            log_path = Path(path_value).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    if handler:
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)


def get_logger(name: str | None = None) -> Logger:
    """Return a module logger with sensible defaults."""
    return logging.getLogger(name if name else "show_scribe")


def set_log_level(level: str | int) -> None:
    """Set the log level on the root logger."""
    logging.getLogger().setLevel(_coerce_level(level))


def _coerce_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str) and level:
        try:
            return logging._nameToLevel[level.upper()]
        except KeyError:
            return logging.INFO
    return logging.INFO
