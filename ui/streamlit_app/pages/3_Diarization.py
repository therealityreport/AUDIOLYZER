"""Diarization stage UI."""

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

import json
import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

AGGRESSIVE_SPLIT_OVERRIDES = {
    "min_duration_on": 0.18,
    "min_duration_off": 0.12,
    "onset": 0.12,
    "offset": 0.06,
    "overlap_threshold": 0.20,
}

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (
    export_diarization_segments,
    get_stage_status,
    resolve_artifact_path,
    run_diarization,
)  # noqa: E402
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Diarization", layout="wide")
active_label, active_dir, _ = render_global_sidebar()
flash = st.session_state.pop("diarization_flash", None)
if flash:
    status = flash.get("status", "success")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)
st.title("Diarization")
if not active_dir:
    st.info("Select an episode to run diarization.")
    st.stop()
st.caption(f"Active episode: **{active_label or active_dir.name}**")


def _resolve_active_audio_path(episode_dir: _Path) -> _Path | None:
    for key in ("audio_enhanced", "audio_vocals", "audio_processed"):
        path = resolve_artifact_path(episode_dir, key)
        if path:
            return path
    return None


active_audio_path = _resolve_active_audio_path(active_dir)
if active_audio_path:
    st.caption(f"Active audio: `{active_audio_path}`")
else:
    st.caption("Active audio: unavailable — run CREATE AUDIO.")
last_log = st.session_state.get("diarization_last_log")
if isinstance(last_log, dict) and last_log.get("episode_dir") == str(active_dir):
    with st.expander("Last diarization run", expanded=False):
        mode = last_log.get("mode")
        if isinstance(mode, str):
            st.caption(f"Speaker-change sensitivity: {mode}")
        if last_log.get("stdout"):
            st.code(last_log["stdout"], language="bash")
        if last_log.get("stderr"):
            st.code(last_log["stderr"], language="bash")
status = get_stage_status(active_dir)
# Check for both legacy and dual transcription artifacts
transcript_ready = status.get("transcript_raw") or (
    status.get("transcript_raw_vocals") and status.get("transcript_raw_mix")
)
if not transcript_ready:
    st.warning("Transcription artifacts not found. Complete transcription before diarization.")

default_mode = st.session_state.get("diarization_sensitivity", "Aggressive")
mode_index = 1 if default_mode == "Aggressive" else 0
mode_choice = st.radio(
    "Speaker change sensitivity",
    ("Balanced", "Aggressive (split at interruptions)"),
    index=mode_index,
    help="Aggressive mode lowers Pyannote's speech-duration thresholds so segments stop as soon as a new voice appears.",
)
is_aggressive = "Aggressive" in mode_choice
st.session_state["diarization_sensitivity"] = "Aggressive" if is_aggressive else "Balanced"
override_params = AGGRESSIVE_SPLIT_OVERRIDES if is_aggressive else {}

trigger = st.button(
    "Run Diarization",
    type="primary",
    disabled=not transcript_ready,
)
if trigger:
    status_indicator = st.status("Queued diarization run…", expanded=True)
    progress_bar = st.progress(0.0)
    status_indicator.write(
        "Preparing diarization job (aggressive speaker-change detection)…"
        if is_aggressive
        else "Preparing diarization job…"
    )
    progress_bar.progress(0.25)
    result = run_diarization(active_dir, overrides=override_params or None)
    progress_bar.progress(0.95)
    st.session_state["diarization_last_log"] = {
        "episode_dir": str(active_dir),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "mode": st.session_state.get("diarization_sensitivity"),
    }
    if result.returncode == 0:
        status_indicator.write(result.stdout or "Diarization completed.")
        status_indicator.update(label="Diarization finished.", state="complete")
        progress_bar.progress(1.0)
        if result.stdout:
            st.code(result.stdout, language="bash")
        st.session_state["diarization_flash"] = {
            "status": "success",
            "message": result.stdout or "Diarization completed.",
        }
    else:
        detail = result.stderr or result.stdout or "Diarization failed."
        status_indicator.write(detail)
        status_indicator.update(label="Diarization failed.", state="error")
        progress_bar.progress(0.05)
        st.code(detail, language="bash")
        st.session_state["diarization_flash"] = {
            "status": "error",
            "message": result.stderr or "Diarization failed.",
        }
    st.rerun()
diarization_path = resolve_artifact_path(active_dir, "diarization")
if diarization_path and diarization_path.exists():
    render_artifact_badge(diarization_path, diarization_path.name)
    try:
        payload = json.loads(diarization_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Unable to parse diarization JSON.")
    else:
        segments = payload.get("segments", [])
        preview = segments[:10]
        st.subheader("Speaker Segments")
        if preview:
            rows = [
                {
                    "id": segment.get("id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "speaker": segment.get("speaker"),
                    "confidence": segment.get("confidence"),
                }
                for segment in preview
            ]
            st.table(rows)
        else:
            st.caption("Diarization file present but no segments found.")
else:
    st.caption("Run diarization to populate diarization artifacts.")

st.divider()
st.subheader("Export Segments For Review")
st.caption(
    "Create per-segment audio clips that automatically stop at speaker changes, "
    "so you can assign names in the next step."
)
export_cols = st.columns(2)
source_label = export_cols[0].selectbox(
    "Source audio",
    options=[
        "Enhanced (mix, with background)",
        "Enhanced (vocals, no background)",
        "Raw Extracted",
    ],
    index=0,
)
# Map selection to artifact key
if "vocals" in source_label:
    source_key = "audio_enhanced_vocals"
elif "mix" in source_label:
    source_key = "audio_enhanced_mix"
else:
    source_key = "audio_extracted"
export_btn = export_cols[1].button(
    "Export Review Segments", disabled=not (diarization_path and diarization_path.exists())
)
if export_btn:
    result = export_diarization_segments(active_dir, source_audio_key=source_key)
    if result.returncode == 0:
        st.success(result.stdout)
    else:
        st.error(result.stderr or "Export failed.")
