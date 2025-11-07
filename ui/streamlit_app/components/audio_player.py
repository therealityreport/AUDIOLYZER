"""Streamlit audio player wrapper."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(audio_path: str | Path, *, label: str | None = None) -> None:
    """Render an inline audio player for the given file path."""

    path = Path(audio_path)
    if not path.exists():
        st.warning(f"Audio file missing: {path}")
        return

    if label:
        st.caption(label)

    try:
        audio_bytes = path.read_bytes()
    except OSError as exc:  # pragma: no cover - streamlit feedback only
        st.error(f"Unable to load audio: {exc}")
        return

    st.audio(audio_bytes, format="audio/wav")
