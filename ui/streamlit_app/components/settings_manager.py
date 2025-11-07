"""User-specific configuration helpers for the Streamlit UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "SettingsManager",
    "PROJECT_ROOT",
    "REPO_SHOWS_ROOT",
    "REPO_VOICE_BANK_ROOT",
    "LEGACY_SHOWS_ROOT",
    "LEGACY_VOICE_BANK_ROOT",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_SHOWS_ROOT = PROJECT_ROOT / "data" / "shows"
REPO_VOICE_BANK_ROOT = PROJECT_ROOT / "data" / "voice_bank"
LEGACY_SHOWS_ROOT = Path("~/Documents/VoiceTranscriptTool/shows").expanduser()
LEGACY_VOICE_BANK_ROOT = Path("~/Documents/VoiceTranscriptTool/voice_bank").expanduser()

SETTINGS_DIR = Path.home() / ".show_scribe"
SETTINGS_PATH = SETTINGS_DIR / "ui_settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "shows_root": str(REPO_SHOWS_ROOT),
    "voice_bank_path": str(REPO_VOICE_BANK_ROOT),
    "default_preset": "configs/reality_tv.yaml",
    "default_show_config": None,
}


def _normalise_path(candidate: str | Path) -> Path:
    """Expand, absolutise, and resolve a candidate path."""

    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve()
    except OSError:
        path = path.expanduser()
    return path


def _within_project(path: Path) -> bool:
    """Return True when ``path`` lives under the project root."""

    try:
        path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return True


class SettingsManager:
    """File-backed storage for Streamlit UI preferences."""

    @classmethod
    def load_settings(cls) -> dict[str, Any]:
        """Load persisted settings merged with defaults."""

        settings = dict(DEFAULT_SETTINGS)
        if SETTINGS_PATH.exists():
            try:
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            if isinstance(raw, dict):
                settings.update(raw)
        migrated = cls._apply_migrations(settings)
        if migrated:
            try:
                cls.save_settings(settings)
            except OSError:
                pass
        return settings

    @classmethod
    def save_settings(cls, settings: dict[str, Any]) -> None:
        """Persist settings to disk, ensuring defaults remain present."""

        payload = dict(DEFAULT_SETTINGS)
        for key, value in (settings or {}).items():
            if key in DEFAULT_SETTINGS:
                payload[key] = value

        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = SETTINGS_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(SETTINGS_PATH)

    @staticmethod
    def _apply_migrations(settings: dict[str, Any]) -> bool:
        """Rewrite legacy defaults to the new in-repo paths."""

        migrated = False

        shows_root = settings.get("shows_root")
        if isinstance(shows_root, str):
            try:
                expanded = Path(shows_root).expanduser()
            except Exception:
                expanded = REPO_SHOWS_ROOT
            legacy_resolved = LEGACY_SHOWS_ROOT
            try:
                legacy_resolved = LEGACY_SHOWS_ROOT.resolve()
            except OSError:
                pass
            try:
                resolved = expanded.resolve()
            except OSError:
                resolved = expanded
            if resolved == legacy_resolved or not _within_project(resolved):
                settings["shows_root"] = str(REPO_SHOWS_ROOT)
                migrated = True

        voice_bank = settings.get("voice_bank_path")
        if isinstance(voice_bank, str):
            try:
                expanded = Path(voice_bank).expanduser()
            except Exception:
                expanded = REPO_VOICE_BANK_ROOT
            legacy_voice = LEGACY_VOICE_BANK_ROOT
            try:
                legacy_voice = LEGACY_VOICE_BANK_ROOT.resolve()
            except OSError:
                pass
            try:
                resolved = expanded.resolve()
            except OSError:
                resolved = expanded
            if resolved == legacy_voice or not _within_project(resolved):
                settings["voice_bank_path"] = str(REPO_VOICE_BANK_ROOT)
                migrated = True

        return migrated

    @classmethod
    def get_shows_root(cls) -> Path:
        """Return the configured shows root directory with environment overrides."""

        settings = cls.load_settings()
        env_override = os.getenv("SHOWS_ROOT")

        if env_override:
            candidate_path = Path(env_override).expanduser()
        else:
            stored = settings.get("shows_root")
            candidate = (
                stored
                if isinstance(stored, str) and stored.strip()
                else DEFAULT_SETTINGS["shows_root"]
            )
            candidate_path = _normalise_path(candidate)
            if not _within_project(candidate_path):
                candidate_path = REPO_SHOWS_ROOT.resolve()

        if env_override:
            candidate_path = _normalise_path(candidate_path)

        try:
            candidate_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        return candidate_path
