"""Initial transcript creation UI with dual ASR workflow."""

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
from datetime import datetime, timezone
from subprocess import CompletedProcess

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (  # noqa: E402
    create_draft_transcript,
    get_stage_status,
    resolve_artifact_path,
    run_dual_alignment,
    run_dual_transcription,
)
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Initial Transcripts", layout="wide")
active_label, active_dir, _ = render_global_sidebar()

flash = st.session_state.pop("transcription_flash", None)
if flash:
    status = flash.get("status", "success")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)

st.title("Initial Transcripts")
if not active_dir:
    st.info("Select an episode to run transcription.")
    st.stop()

st.caption(f"Active episode: **{active_label or active_dir.name}**")

# Check for enhanced audio artifacts
vocals_path = resolve_artifact_path(active_dir, "audio_enhanced_vocals")
mix_path = resolve_artifact_path(active_dir, "audio_enhanced_mix")

audio_ready = bool(vocals_path and vocals_path.exists() and mix_path and mix_path.exists())

st.subheader("Workflow: Dual Transcription → Diarization → Draft")
st.write(
    "This workflow creates two ASR transcripts (from vocals and mix), "
    "runs diarization on vocals, aligns both transcripts, and generates "
    "draft transcripts with numbered speakers (SPEAKER_00, SPEAKER_01, etc.)."
)

if not audio_ready:
    st.warning(
        "⚠️ Missing enhanced audio artifacts. Run **CREATE AUDIO** first to generate "
        "`audio_enhanced_vocals.wav` and `audio_enhanced_mix.wav`."
    )
else:
    st.success(
        f"✓ Enhanced vocals: `{vocals_path.name if vocals_path else 'N/A'}`\n\n"
        f"✓ Enhanced mix: `{mix_path.name if mix_path else 'N/A'}`"
    )

# Show existing artifacts
st.subheader("Artifacts")
cols = st.columns(3)

with cols[0]:
    st.caption("**Raw Transcripts**")
    vocals_trans = resolve_artifact_path(active_dir, "transcript_raw_vocals")
    mix_trans = resolve_artifact_path(active_dir, "transcript_raw_mix")
    if vocals_trans and vocals_trans.exists():
        render_artifact_badge(vocals_trans, "vocals.json")
    else:
        st.caption("❌ transcript_raw.vocals.json")
    if mix_trans and mix_trans.exists():
        render_artifact_badge(mix_trans, "mix.json")
    else:
        st.caption("❌ transcript_raw.mix.json")

with cols[1]:
    st.caption("**Diarization & Alignment**")
    diar_path = resolve_artifact_path(active_dir, "diarization")
    aligned_vocals = resolve_artifact_path(active_dir, "aligned_vocals")
    aligned_mix = resolve_artifact_path(active_dir, "aligned_mix")
    if diar_path and diar_path.exists():
        render_artifact_badge(diar_path, "diarization.json")
    else:
        st.caption("❌ diarization.json")
    if aligned_vocals and aligned_vocals.exists():
        render_artifact_badge(aligned_vocals, "aligned_vocals.jsonl")
    else:
        st.caption("❌ aligned_vocals.jsonl")
    if aligned_mix and aligned_mix.exists():
        render_artifact_badge(aligned_mix, "aligned_mix.jsonl")
    else:
        st.caption("❌ aligned_mix.jsonl")

with cols[2]:
    st.caption("**Draft Transcripts**")
    draft_txt = resolve_artifact_path(active_dir, "transcript_draft_txt")
    draft_srt = resolve_artifact_path(active_dir, "transcript_draft_srt")
    if draft_txt and draft_txt.exists():
        render_artifact_badge(draft_txt, "draft.txt")
    else:
        st.caption("❌ transcript_draft.txt")
    if draft_srt and draft_srt.exists():
        render_artifact_badge(draft_srt, "draft.srt")
    else:
        st.caption("❌ transcript_draft.srt")

st.divider()


def _trigger_rerun() -> None:
    try:
        st.rerun()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - compatibility fallback
        st.rerun()


# Workflow execution
st.subheader("Run Workflow")

source_choice = st.radio(
    "Draft transcript source",
    options=["vocals", "mix"],
    index=0,
    help="Choose which aligned transcript to use for the draft",
)

trigger = st.button(
    "Create Initial Transcripts",
    type="primary",
    disabled=not audio_ready,
    help="Runs: Dual ASR → Diarization → Alignment → Draft generation",
)

if trigger:
    status_indicator = st.status("Starting dual transcription workflow…", expanded=True)
    progress_bar = st.progress(0.0)

    progress_entries: list[dict[str, object]] = []

    def _report_progress(message: str, fraction: float | None = None) -> None:
        if fraction is not None:
            bounded = max(0.0, min(1.0, float(fraction)))
            percent = int(round(bounded * 100))
            progress_bar.progress(bounded)
            status_indicator.write(f"{percent}% · {message}")
        else:
            status_indicator.write(message)

    try:
        # Step 1: Dual transcription
        _report_progress("Running dual ASR transcription…", 0.0)
        trans_result = run_dual_transcription(active_dir, progress=_report_progress)

        if trans_result.returncode != 0:
            st.error(f"Dual transcription failed: {trans_result.stderr}")
            status_indicator.update(label="Dual transcription failed", state="error")
            st.stop()

        _report_progress("Dual transcription complete", 0.33)

        # Step 2: Dual alignment (includes diarization)
        _report_progress("Running diarization and dual alignment…", 0.35)
        align_result = run_dual_alignment(active_dir, progress=_report_progress)

        if align_result.returncode != 0:
            st.error(f"Alignment failed: {align_result.stderr}")
            status_indicator.update(label="Alignment failed", state="error")
            st.stop()

        _report_progress("Alignment complete", 0.66)

        # Step 3: Create draft transcripts
        _report_progress(f"Creating draft transcripts from {source_choice}…", 0.7)
        draft_result = create_draft_transcript(
            active_dir, source=source_choice, progress=_report_progress
        )

        if draft_result.returncode != 0:
            st.error(f"Draft creation failed: {draft_result.stderr}")
            status_indicator.update(label="Draft creation failed", state="error")
            st.stop()

        _report_progress("Workflow complete", 1.0)
        status_indicator.update(label="Workflow completed successfully", state="complete")
        progress_bar.progress(1.0)

        st.success(
            f"✓ Dual transcription: {trans_result.stdout}\n\n"
            f"✓ Alignment: {align_result.stdout}\n\n"
            f"✓ Draft: {draft_result.stdout}"
        )

        st.session_state["transcription_flash"] = {
            "status": "success",
            "message": "Initial transcripts workflow completed successfully!",
        }

    except Exception as exc:
        st.error(f"Workflow failed: {exc}")
        status_indicator.update(label="Workflow failed", state="error")
        st.session_state["transcription_flash"] = {
            "status": "error",
            "message": f"Workflow failed: {exc}",
        }

    _trigger_rerun()

# Preview draft transcript
st.divider()
st.subheader("Draft Transcript Preview")

draft_txt = resolve_artifact_path(active_dir, "transcript_draft_txt")
if draft_txt and draft_txt.exists():
    try:
        draft_content = draft_txt.read_text(encoding="utf-8")
        lines = draft_content.split("\n\n")
        preview = "\n\n".join(lines[:20])
        st.text_area("Draft (first 20 segments)", preview, height=400)

        if len(lines) > 20:
            st.caption(f"Showing 20 of {len(lines)} segments")

        # Show download button
        st.download_button(
            "Download Draft TXT",
            data=draft_content,
            file_name="transcript_draft.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"Failed to read draft: {exc}")
else:
    st.caption("Run the workflow to generate draft transcripts.")

draft_srt = resolve_artifact_path(active_dir, "transcript_draft_srt")
if draft_srt and draft_srt.exists():
    try:
        srt_content = draft_srt.read_text(encoding="utf-8")
        st.download_button(
            "Download Draft SRT",
            data=srt_content,
            file_name="transcript_draft.srt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"Failed to read SRT: {exc}")
