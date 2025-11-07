"""Helpers for generating canonical filenames for episode artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .paths import PathsConfig

__all__ = [
    "EpisodeDescriptor",
    "build_artifact_filename",
    "build_episode_id",
    "resolve_artifact_path",
]


_SHOW_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ARTIFACT_KEY_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class EpisodeDescriptor:
    """Structured information that uniquely identifies an episode."""

    show_name: str
    season: int
    episode: int
    variant: str | None = None

    @property
    def show_slug(self) -> str:
        """Return the compact show identifier used in filenames."""
        tokens = _SHOW_TOKEN_RE.findall(self.show_name)
        if not tokens:
            raise ValueError(f"Unable to derive show slug from {self.show_name!r}.")
        return "".join(token.title() for token in tokens)

    @property
    def episode_code(self) -> str:
        """Return the canonical episode code ``SxxEyy``."""
        if self.season < 0 or self.episode < 0:
            raise ValueError("Season and episode must be non-negative integers.")
        return f"S{self.season:02d}E{self.episode:02d}"

    @property
    def episode_id(self) -> str:
        """Return the canonical episode identifier used for directories."""
        parts = [self.show_slug, self.episode_code]
        if self.variant:
            parts.append(self.variant)
        return "_".join(parts)


def build_episode_id(descriptor: EpisodeDescriptor) -> str:
    """Return the canonical episode identifier for the descriptor."""
    return descriptor.episode_id


def build_artifact_filename(
    descriptor: EpisodeDescriptor, artifact_key: str, extension: str
) -> str:
    """Return a filename that follows the configured naming convention."""
    if not _ARTIFACT_KEY_RE.match(artifact_key):
        raise ValueError(
            "artifact_key must contain lowercase letters, numbers, and underscores only."
        )
    ext = extension.lstrip(".").lower()
    parts = [descriptor.show_slug, descriptor.episode_code]
    if descriptor.variant:
        parts.append(descriptor.variant)
    parts.append(artifact_key)
    base = "_".join(parts)
    return f"{base}.{ext}"


def resolve_artifact_path(
    paths: PathsConfig,
    descriptor: EpisodeDescriptor,
    artifact_key: str,
    extension: str,
    *,
    ensure_directory: bool = False,
) -> Path:
    """Return the filesystem path where an artifact should be stored."""
    episode_dir = paths.episode_directory(descriptor.show_name, descriptor.episode_id)
    if ensure_directory:
        episode_dir.mkdir(parents=True, exist_ok=True)
    filename = build_artifact_filename(descriptor, artifact_key, extension)
    return episode_dir / filename
