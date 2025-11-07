"""Transcript exports and downloads."""

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

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (  # noqa: E402
    apply_manual_speaker_mapping,
    build_exports,
    get_stage_status,
    read_metadata,
    resolve_artifact_path,
    run_align_and_export,
)
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Speaker Transcript", layout="wide")
active_label, active_dir, _ = render_global_sidebar()
flash = st.session_state.pop("exports_flash", None)
if flash:
    status = flash.get("status", "info")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)
st.title("Speaker Transcript")
if not active_dir:
    st.info("Select an episode to review transcript exports.")
    st.stop()
st.caption(f"Active episode: **{active_label or active_dir.name}**")
status = get_stage_status(active_dir)
align_ready = bool(status.get("transcript_raw") and status.get("diarization"))
speaker_ready = bool(status.get("transcript_final"))
if not align_ready:
    st.warning("Run transcription and diarization before aligning the transcript.")
align_button = st.button(
    "Create Speaker Transcript (Align)",
    type="primary",
    disabled=not align_ready,
)
if align_button:
    result = run_align_and_export(active_dir)
    if result.returncode == 0:
        st.session_state["exports_flash"] = {
            "status": "success",
            "message": result.stdout or "Alignment completed.",
        }
    else:
        st.session_state["exports_flash"] = {
            "status": "error",
            "message": result.stderr or "Alignment failed.",
        }
    st.rerun()
if not speaker_ready:
    st.warning("Speaker identification should be completed before generating exports.")
trigger = st.button(
    "Generate Transcript Exports",
    type="secondary",
    disabled=not speaker_ready,
)
if trigger:
    result = build_exports(active_dir)
    if result.returncode == 0:
        st.session_state["exports_flash"] = {
            "status": "success",
            "message": result.stdout or "Transcript exports generated.",
        }
    else:
        st.session_state["exports_flash"] = {
            "status": "error",
            "message": result.stderr or "Export generation failed.",
        }
    st.rerun()

# Speaker Mapping Section
st.divider()
st.subheader("Update Speaker Names")
st.caption("Bulk replace speaker labels in the transcript (e.g., SPEAKER_00 → Lisa Rinna)")

if speaker_ready:
    transcript_path = resolve_artifact_path(active_dir, "transcript_final")
    if transcript_path and transcript_path.exists():
        import json as _json

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = _json.load(f)

            # Extract unique speakers
            segments = transcript_data.get("segments", [])
            speakers_set = set()
            for seg in segments:
                if isinstance(seg, dict):
                    speaker = seg.get("speaker")
                    if speaker and isinstance(speaker, str):
                        speakers_set.add(speaker.strip())

            speakers = sorted(speakers_set)

            if speakers:
                st.caption(f"Current speakers in transcript: {', '.join(speakers)}")

                # Create mapping inputs
                mapping_inputs = {}
                cols_per_row = 3
                for row_start in range(0, len(speakers), cols_per_row):
                    row_speakers = speakers[row_start : row_start + cols_per_row]
                    cols = st.columns(len(row_speakers))
                    for idx, speaker in enumerate(row_speakers):
                        with cols[idx]:
                            mapping_inputs[speaker] = st.text_input(
                                f"{speaker} →",
                                value="",
                                placeholder="New name (e.g., Lisa Rinna)",
                                key=f"speaker_map_{speaker}",
                            )

                if st.button("Apply Speaker Mapping", type="primary"):
                    mapping = {k: v.strip() for k, v in mapping_inputs.items() if v and v.strip()}
                    if not mapping:
                        st.warning("Provide at least one new speaker name before applying.")
                    else:
                        result = apply_manual_speaker_mapping(active_dir, mapping)
                        if result.returncode == 0:
                            st.session_state["exports_flash"] = {
                                "status": "success",
                                "message": result.stdout or "Speaker mapping applied successfully.",
                            }
                            st.rerun()
                        else:
                            st.error(result.stderr or "Failed to apply speaker mapping.")
            else:
                st.caption("No speakers found in transcript.")
        except Exception as e:
            st.error(f"Failed to load transcript: {e}")
else:
    st.caption("Complete speaker alignment first.")

st.divider()
st.subheader("Export Files")
metadata = read_metadata(active_dir) or {}
prefix = metadata.get("artifact_prefix") or active_dir.name
transcripts_dir = active_dir / "transcripts"


def _find_export_file(filename_suffix: str, legacy_filename: str) -> Path | None:
    prefixed = transcripts_dir / f"{prefix}_{filename_suffix}"
    if prefixed.exists():
        return prefixed
    legacy = active_dir / legacy_filename
    if legacy.exists():
        return legacy
    return None


export_candidates = {
    "Transcript (TXT)": _find_export_file("transcript_final.txt", "transcript_final.txt"),
    "Transcript (SRT)": _find_export_file("transcript_final.srt", "transcript_final.srt"),
    "Transcript (JSON)": resolve_artifact_path(active_dir, "transcript_final")
    or _find_export_file("transcript_final.json", "transcript_final.json"),
}
for label, path in export_candidates.items():
    if path and path.exists():
        render_artifact_badge(path, path.name)
        with path.open("rb") as handle:
            data = handle.read()
        st.download_button(
            label=f"Download {label}",
            data=data,
            file_name=path.name,
            mime="text/plain" if path.suffix != ".json" else "application/json",
        )
        if path.suffix == ".txt":
            st.subheader("Transcript Preview")
            preview_text = data.decode("utf-8", errors="ignore")
            st.text(preview_text[:2000] + ("\n…" if len(preview_text) > 2000 else ""))
    else:
        st.caption(f"{label} not found yet.")
