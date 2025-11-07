"""Shared layout helpers for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from show_scribe.ui_ops import read_metadata

from .episode_badge import render_episode_badge
from .episode_picker import render_episode_picker
from .stage_checklist import render_stage_checklist


def render_global_sidebar() -> tuple[str | None, Path | None, Mapping[str, Any]]:
    """Render the shared sidebar controls and return current selection."""

    with st.sidebar:
        st.header("Episode Selection")
        label, episode_dir = render_episode_picker()

        metadata: Mapping[str, Any] = {}
        if episode_dir:
            metadata = read_metadata(episode_dir) or {}
            render_episode_badge(episode_dir, metadata)
            render_stage_checklist(episode_dir)
        else:
            st.caption("Create an episode on the Process Episode page to populate this list.")

    return label, episode_dir, metadata
