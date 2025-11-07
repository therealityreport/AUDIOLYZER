"""Helpers for deriving canonical filesystem paths from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PathsConfig", "build_paths"]


def _normalize_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    """Return an absolute path, interpreting relative paths from ``relative_to``."""
    path = Path(value)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.expanduser().resolve()


@dataclass(slots=True)
class PathsConfig:
    """Resolved filesystem paths used throughout the project."""

    project_root: Path
    data_root: Path
    output_root: Path
    cache_dir: Path
    temp_dir: Path
    models_dir: Path
    voice_bank_db: Path
    logs_dir: Path

    def ensure_directories(self, *, include_outputs: bool = True) -> None:
        """Create directories that should always exist."""
        required = [
            self.data_root,
            self.cache_dir,
            self.temp_dir,
            self.models_dir,
            self.voice_bank_db.parent,
            self.logs_dir,
        ]
        if include_outputs:
            required.append(self.output_root)

        for directory in required:
            directory.mkdir(parents=True, exist_ok=True)

    # Convenience helpers -------------------------------------------------
    def show_root(self, show_name: str) -> Path:
        """Return the directory for a particular show."""
        return self.data_root / "shows" / show_name

    def show_config_path(self, show_name: str) -> Path:
        """Path to the JSON configuration for a show."""
        return self.show_root(show_name) / "show_config.json"

    def episode_directory(self, show_name: str, episode_id: str) -> Path:
        """Directory used to store processed episode artifacts."""
        return self.show_root(show_name) / "episodes" / episode_id

    def voice_bank_backup_dir(self) -> Path:
        """Directory where automatic backups of the voice bank are written."""
        return self.data_root / "voice_bank" / "backups"


def build_paths(config: Mapping[str, object]) -> PathsConfig:
    """Construct a :class:`PathsConfig` from the parsed configuration."""
    paths_section = config.get("paths")
    if not isinstance(paths_section, Mapping):
        raise ValueError("Configuration is missing the 'paths' section.")

    project_root_raw = paths_section.get("project_root", ".")
    project_root = _normalize_path(project_root_raw)

    def resolve(key: str) -> Path:
        raw_value = paths_section.get(key)
        if raw_value is None:
            raise ValueError(f"Configuration 'paths.{key}' is required.")
        return _normalize_path(raw_value, relative_to=project_root)

    return PathsConfig(
        project_root=project_root,
        data_root=resolve("data_root"),
        output_root=resolve("output_root"),
        cache_dir=resolve("cache_dir"),
        temp_dir=resolve("temp_dir"),
        models_dir=resolve("models_dir"),
        voice_bank_db=resolve("voice_bank_db"),
        logs_dir=resolve("logs_dir"),
    )
