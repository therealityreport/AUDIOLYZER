"""Global episode selector shared across Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from .settings_manager import SettingsManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SHOWS_ROOT = Path("~/Documents/VoiceTranscriptTool/shows").expanduser()
SESSION_DIR_KEY = "active_episode_dir"
SESSION_LABEL_KEY = "active_episode_label"


def _iter_metadata_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    pattern = root.glob("*/episodes/*/metadata.json")
    return sorted(pattern)


def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _format_label(episode_dir: Path, metadata: dict[str, Any] | None) -> tuple[str, Path]:
    episode_id = metadata.get("episode_id") if isinstance(metadata, dict) else None
    show_slug = metadata.get("show_slug") if isinstance(metadata, dict) else None

    if not episode_id:
        episode_id = episode_dir.name

    if not show_slug:
        try:
            show_slug = episode_dir.parents[1].name
        except IndexError:
            show_slug = "Unknown"

    label = f"{show_slug} · {episode_id}"
    return label, episode_dir


def _collect_episodes() -> list[tuple[str, Path]]:
    shows_root = SettingsManager.get_shows_root()
    metadata_files = _iter_metadata_files(shows_root)
    legacy_files: list[Path] = []
    legacy_root = LEGACY_SHOWS_ROOT
    try:
        if legacy_root.exists() and legacy_root.resolve() != shows_root.resolve():
            legacy_files = _iter_metadata_files(legacy_root)
    except OSError:
        pass

    entries: list[tuple[str, Path]] = []
    seen = set()
    for metadata_path in metadata_files + legacy_files:
        episode_dir = metadata_path.parent
        metadata = _load_metadata(metadata_path)
        label, directory = _format_label(episode_dir, metadata)
        if label in seen:
            suffix = directory.name
            label = f"{label} ({suffix})"
        seen.add(label)
        entries.append((label, directory))

    entries.sort(key=lambda entry: entry[0].lower())
    return entries


def list_available_episodes() -> list[tuple[str, Path]]:
    """Return discovered episodes without mutating session state."""

    return _collect_episodes()


def render_episode_picker() -> tuple[str | None, Path | None]:
    """Render sidebar selector and update session state."""

    episodes = _collect_episodes()

    if not episodes:
        st.info("No episodes found under the configured shows directory yet.")
        st.session_state.pop(SESSION_DIR_KEY, None)
        st.session_state.pop(SESSION_LABEL_KEY, None)
        return None, None

    labels = [label for label, _ in episodes]
    label_to_dir = {label: directory for label, directory in episodes}

    default_label = st.session_state.get(SESSION_LABEL_KEY)
    if default_label not in label_to_dir:
        default_label = labels[0]

    index = labels.index(default_label)

    selection = st.selectbox(
        "Episode",
        options=labels,
        index=index,
        key="global_episode_picker",
    )

    directory = label_to_dir.get(selection)

    st.session_state[SESSION_LABEL_KEY] = selection
    st.session_state[SESSION_DIR_KEY] = str(directory) if directory else None

    return selection, directory
