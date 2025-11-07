"""Compact stage progress indicator for an episode."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from show_scribe.ui_ops import get_stage_status

STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("Audio Extraction", ("audio_processed", "audio_extracted")),
    ("Transcription", ("transcript_raw",)),
    ("Diarization", ("diarization",)),
    ("Bleep Detection", ("bleeps",)),
    ("Speaker ID", ("transcript_final",)),
    ("Exports", ("transcript_final",)),
    ("Analytics", ("analytics",)),
]


def render_stage_checklist(episode_dir: Path) -> None:
    """Render pipeline stage completion checklist."""

    if not episode_dir:
        st.info("Select an episode to view stage progress.")
        return

    status = get_stage_status(episode_dir)
    completion = []
    for _, keys in STAGES:
        done = any(status.get(key, False) for key in keys)
        completion.append(done)

    st.markdown("#### Pipeline Progress")
    for index, (label, _) in enumerate(STAGES):
        done = completion[index]
        if done:
            icon = "✓"
            text = label
        else:
            if all(completion[:index]):
                icon = "⧗"
                text = f"{label} (ready)"
            else:
                icon = "○"
                text = label
        st.markdown(f"{icon} {text}")

    if all(completion):
        st.success("All stages complete.")
    else:
        try:
            next_index = completion.index(False)
            next_label = STAGES[next_index][0]
        except ValueError:
            next_label = None
        if next_label:
            st.caption(f"Next stage: {next_label}")
            if st.button(f"Resume {next_label}", key="resume_stage_button"):
                st.session_state["stage_resume_target"] = next_label
