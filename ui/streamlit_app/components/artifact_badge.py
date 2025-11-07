"""Lightweight badge widget showing artifact availability."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_artifact_badge(artifact_path: Path, label: str) -> None:
    """Render coloured status badge for an artifact path."""

    path = artifact_path.expanduser()
    if path.exists():
        color = "#1d8348"  # green
        state = "Available"
    elif path.parent.exists():
        color = "#b9770e"  # amber
        state = "Pending"
    else:
        color = "#c0392b"  # red
        state = "Missing"

    badge = f"<span style='background-color:{color};color:white;padding:0.1rem 0.5rem;border-radius:0.75rem;font-size:0.8rem;'>{state}</span>"
    st.markdown(f"{badge} &nbsp; **{label}**", unsafe_allow_html=True)

    st.caption(f"{path}")
