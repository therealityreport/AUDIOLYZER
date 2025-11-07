"""Episode information badge."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import streamlit as st


def _human_ts(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M")


def render_episode_badge(episode_dir: Path, metadata: Mapping[str, Any] | None) -> None:
    """Display high-level metadata for the active episode."""

    metadata = metadata or {}
    episode_dir = episode_dir.expanduser()

    show_slug = str(
        metadata.get("show_slug") or episode_dir.parents[1].name
        if len(episode_dir.parents) > 1
        else "Unknown"
    )
    episode_id = str(metadata.get("episode_id") or episode_dir.name)
    preset = str(metadata.get("preset") or "default")
    created = _human_ts(metadata.get("created_at"))
    updated = _human_ts(metadata.get("updated_at"))

    st.markdown(f"### {show_slug} · {episode_id}")
    st.caption(f"Preset `{preset}` · Created {created} · Updated {updated}")
    st.caption(str(episode_dir))
