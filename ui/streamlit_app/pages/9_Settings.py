"""UI for configuring Streamlit-specific settings."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PACKAGE_ROOT = _Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _candidate in (_PROJECT_ROOT, _SRC_DIR):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import shutil
from pathlib import Path

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402
from ui.streamlit_app.components.settings_manager import (  # noqa: E402
    LEGACY_SHOWS_ROOT,
    SettingsManager,
)

st.set_page_config(page_title="Settings", layout="wide")
render_global_sidebar()
st.title("Settings")
settings = SettingsManager.load_settings()
with st.form("paths_form", clear_on_submit=False):
    st.subheader("Paths Configuration")
    shows_root_input = st.text_input(
        "Shows root directory", value=str(settings.get("shows_root", ""))
    )
    voice_bank_input = st.text_input(
        "Voice bank path", value=str(settings.get("voice_bank_path", ""))
    )
    default_preset_input = st.text_input(
        "Default preset file", value=str(settings.get("default_preset", ""))
    )
    default_show_config_input = st.text_input(
        "Default show config (optional)",
        value=str(settings.get("default_show_config") or ""),
    )
    submitted = st.form_submit_button("Save Settings", type="primary")
if submitted:
    updated = dict(settings)
    updated.update(
        {
            "shows_root": shows_root_input.strip() or settings["shows_root"],
            "voice_bank_path": voice_bank_input.strip() or settings["voice_bank_path"],
            "default_preset": default_preset_input.strip() or settings["default_preset"],
            "default_show_config": default_show_config_input.strip() or None,
        }
    )
    SettingsManager.save_settings(updated)
    st.success("Settings saved.")
    st.rerun()
st.subheader("Environment Check")


def _status_badge(passed: bool) -> str:
    return "✅" if passed else "⚠️"


ffmpeg_available = shutil.which("ffmpeg") is not None
st.markdown(f"{_status_badge(ffmpeg_available)} FFmpeg available")
models_dir = ROOT / "data" / "models"
whisper_dir = models_dir / "whisper"
pyannote_dir = models_dir / "pyannote"
whisper_cached = whisper_dir.exists() and any(whisper_dir.iterdir())
pyannote_cached = pyannote_dir.exists() and any(pyannote_dir.iterdir())
st.markdown(f"{_status_badge(whisper_cached)} Whisper models cached at `{whisper_dir}`")
st.markdown(f"{_status_badge(pyannote_cached)} Pyannote models cached at `{pyannote_dir}`")
st.caption(
    "Use the setup scripts under `scripts/setup/` to download required models "
    "(e.g., `python scripts/setup/download_models.py`)."
)
shows_root = SettingsManager.get_shows_root()
legacy_root = LEGACY_SHOWS_ROOT
try:
    legacy_resolved = legacy_root.resolve()
except OSError:
    legacy_resolved = legacy_root
try:
    shows_root_resolved = shows_root.resolve()
except OSError:
    shows_root_resolved = shows_root
if legacy_root.exists() and legacy_resolved != shows_root_resolved:
    st.subheader("Legacy Data Migration")
    st.caption(f"Migrate episodes from `{legacy_root}` to `{shows_root}`.")
    if st.button("Migrate legacy episodes"):
        migrated: list[str] = []
        shows_root.mkdir(parents=True, exist_ok=True)
        for show_dir in sorted(legacy_root.iterdir()):
            if not show_dir.is_dir():
                continue
            destination = shows_root / show_dir.name
            if destination.exists():
                continue
            shutil.copytree(show_dir, destination)
            migrated.append(show_dir.name)
        if migrated:
            st.success(f"Migrated {len(migrated)} shows: {', '.join(migrated)}")
        else:
            st.info("No shows required migration.")
