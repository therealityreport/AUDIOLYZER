"""Streamlit landing page for the Show-Scribe multipage UI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root and src directory are importable when Streamlit runs this module.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _candidate in (_PROJECT_ROOT, _SRC_DIR):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from ui.streamlit_app.bootstrap import initialise_paths  # noqa: E402

ROOT, SRC_DIR = initialise_paths()

from ui.streamlit_app.components.episode_picker import list_available_episodes  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402
from ui.streamlit_app.components.settings_manager import SettingsManager  # noqa: E402


st.set_page_config(page_title="Show-Scribe Studio", layout="wide")

active_label, active_dir, _ = render_global_sidebar()

episodes = list_available_episodes()
shows_root = SettingsManager.get_shows_root()

st.title("Show-Scribe Studio")
st.write(
    "Welcome to the revamped Show-Scribe workflow. Use the navigation below to walk through "
    "each stage of the processing pipeline, or jump directly to review pages for an existing episode."
)

cols = st.columns(3)
cols[0].metric("Available Episodes", len(episodes))
unique_shows = {
    directory.parents[1].name for _, directory in episodes if len(directory.parents) > 1
}
cols[1].metric("Tracked Shows", len(unique_shows))
cols[2].metric("Active Episode", active_label or "—")
cols[2].caption(f"Shows root: {shows_root}")

resume_target = st.session_state.get("stage_resume_target")
if resume_target:
    st.info(
        f"Next suggested stage: {resume_target}. Open the corresponding page below to continue."
    )
    st.session_state.pop("stage_resume_target", None)

st.subheader("Stage Navigation")
page_links = [
    ("🎬 Process Episode", "pages/1_Process_Episode.py"),
    ("📝 Initial Transcript", "pages/2_Initial_Transcript.py"),
    ("🧑‍🤝‍🧑 Diarization", "pages/3_Diarization.py"),
    ("🎙️ Voice Bank", "pages/7_Voice_Bank.py"),
    ("📄 Speaker Transcript", "pages/5_Transcript_and_Exports.py"),
    ("📊 Analytics (Later)", "pages/6_Analytics.py"),
    ("🔇 Bleep Review (Later)", "pages/3b_Bleep_Review.py"),
    ("⚙️ Settings", "pages/8_Settings.py"),
]

for label, target in page_links:
    st.page_link(target, label=label)

st.divider()
st.caption(
    "Need a new episode? Start with Process Episode to scaffold inputs and kick off preprocessing. "
    "Metadata and pipeline artifacts live under your configured shows root."
)
