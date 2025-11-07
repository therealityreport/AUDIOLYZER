"""Voice bank browser."""

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

from pathlib import Path

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.storage.db import SQLiteDatabase  # noqa: E402
from show_scribe.storage.voice_bank_manager import VoiceBankManager  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402
from ui.streamlit_app.components.settings_manager import SettingsManager  # noqa: E402

st.set_page_config(page_title="Voice Bank", layout="wide")
render_global_sidebar()
st.title("Voice Bank")
settings = SettingsManager.load_settings()
configured_path = settings.get("voice_bank_path") or ""
voice_bank_root = Path(configured_path).expanduser()
if voice_bank_root.is_dir():
    db_path = voice_bank_root / "voice_bank.sqlite3"
    base_dir = voice_bank_root
else:
    db_path = voice_bank_root
    base_dir = db_path.parent
st.caption(f"Voice bank database: `{db_path}`")
if not db_path.exists():
    st.warning("Voice bank database not found. Configure the path on the Settings page.")
    st.stop()
database = SQLiteDatabase(db_path)
manager = VoiceBankManager(database)
profiles = manager.list_speakers()
if not profiles:
    st.info("No speaker profiles registered yet.")
else:
    st.subheader(f"Registered Speakers ({len(profiles)})")
    records = [
        {
            "Display Name": profile.display_name,
            "Key": profile.key,
            "Aliases": ", ".join(profile.common_aliases),
            "Misspellings": ", ".join(profile.common_misspellings),
            "First Appearance": profile.first_appearance_episode or "",
            "Total Segments": profile.total_segments,
            "Notes": profile.notes or "",
            "Updated": profile.updated_at.isoformat() if profile.updated_at else "",
        }
        for profile in profiles
    ]
    st.dataframe(records, use_container_width=True)

st.divider()
st.subheader("Speaker Samples")
st.caption("Browse and add voice segments per speaker for better identification.")

samples_root = base_dir / "samples"
samples_root.mkdir(parents=True, exist_ok=True)

if profiles:
    names = [p.display_name for p in profiles]
    selected = st.selectbox("Select speaker", options=names, index=0)
    current = next((p for p in profiles if p.display_name == selected), None)
    if current:
        speaker_dir = samples_root / current.key
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # List DB-linked samples (if any)
        st.caption("Samples linked in DB (if available):")
        linked = manager.list_embeddings(current.id) if current.id is not None else []
        if linked:
            cols = st.columns(3)
            for idx, entry in enumerate(linked[:6]):
                with cols[idx % 3]:
                    if entry.audio_sample_path and entry.audio_sample_path.exists():
                        from ui.streamlit_app.components import audio_player as _ap

                        _ap.render(entry.audio_sample_path, label=str(entry.audio_sample_path.name))
        else:
            st.caption("No DB-linked samples.")

        st.caption("Samples folder:")
        sample_files = sorted([p for p in speaker_dir.glob("*.wav") if p.is_file()])
        if sample_files:
            cols = st.columns(3)
            for idx, path in enumerate(sample_files[:9]):
                with cols[idx % 3]:
                    from ui.streamlit_app.components import audio_player as _ap

                    _ap.render(path, label=path.name)
        else:
            st.caption("No files in samples folder yet.")

        st.subheader("Add Sample")
        uploaded = st.file_uploader("Upload WAV sample", type=["wav"])
        if uploaded is not None:
            out_path = speaker_dir / uploaded.name
            out_path.write_bytes(uploaded.getbuffer())
            st.success(f"Saved sample to {out_path}")

st.divider()
st.subheader("Merge Speakers")
st.caption("Coming soon – consolidate duplicate speaker profiles across the voice bank.")
