"""Speaker identification and manual review UI."""

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
from collections import Counter, defaultdict

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (  # noqa: E402
    assign_segment_speakers,
    get_stage_status,
    resolve_artifact_path,
    run_speaker_id,
    apply_manual_speaker_mapping,
)
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components import audio_player  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Speaker ID & Review", layout="wide")
active_label, active_dir, _ = render_global_sidebar()
flash = st.session_state.pop("speaker_flash", None)
if flash:
    status = flash.get("status", "info")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)
st.title("Speaker Identification & Review")
if not active_dir:
    st.info("Select an episode to review speaker identification.")
    st.stop()
st.caption(f"Active episode: **{active_label or active_dir.name}**")
status = get_stage_status(active_dir)
diarization_ready = status.get("diarization")
if not diarization_ready:
    st.warning("Diarization results required before running speaker identification.")
trigger = st.button(
    "Run Speaker Identification",
    type="primary",
    disabled=not diarization_ready,
)
if trigger:
    result = run_speaker_id(active_dir)
    if result.returncode == 0:
        st.session_state["speaker_flash"] = {
            "status": "success",
            "message": result.stdout or "Speaker identification completed.",
        }
    else:
        st.session_state["speaker_flash"] = {
            "status": "error",
            "message": result.stderr or "Speaker identification failed.",
        }
    st.rerun()
transcript_path = resolve_artifact_path(active_dir, "transcript_final")
if transcript_path and transcript_path.exists():
    render_artifact_badge(transcript_path, transcript_path.name)
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Unable to parse final transcript JSON.")
    else:
        st.subheader("Speaker Distribution")
        segments = payload.get("segments", payload.get("data", []))
        speakers = [
            segment.get("speaker", "UNKNOWN") for segment in segments if isinstance(segment, dict)
        ]
        if speakers:
            counts = Counter(speakers)
            cols = st.columns(3)
            for index, (name, count) in enumerate(counts.most_common()):
                cols[index % 3].metric(name, count)
        else:
            st.caption("No segment speaker assignments found.")
        unknown_segments = [
            segment
            for segment in segments
            if isinstance(segment, dict)
            and str(segment.get("speaker", "")).upper().startswith("UNKNOWN")
        ]
        st.subheader("Segments Requiring Review")
        if unknown_segments:
            preview = [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                }
                for segment in unknown_segments[:20]
            ]
            st.table(preview)
            with st.expander("Manual Speaker Mapping", expanded=True):
                st.caption("Map diarization clusters to display names and apply to the transcript.")

                # Build unique set of original cluster labels
                original_labels: list[str] = []
                seen: set[str] = set()
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    meta = seg.get("metadata") or {}
                    original = meta.get("original_speaker") or seg.get("speaker")
                    if not isinstance(original, str):
                        continue
                    key = original.strip()
                    if not key or key in seen:
                        continue
                    # Focus on clusters/unknowns by default
                    if key.upper().startswith("SPEAKER_") or key.upper().startswith("UNKNOWN"):
                        original_labels.append(key)
                        seen.add(key)

                if not original_labels:
                    st.caption("No unmapped speaker clusters detected.")
                else:
                    cols_per_row = 3
                    inputs: dict[str, str] = {}
                    for row_start in range(0, len(original_labels), cols_per_row):
                        row = original_labels[row_start : row_start + cols_per_row]
                        cols = st.columns(len(row))
                        for idx, label in enumerate(row):
                            with cols[idx]:
                                inputs[label] = st.text_input(
                                    f"{label} →",
                                    value="",
                                    placeholder="Enter display name (e.g., Lisa)",
                                    key=f"map_{label}",
                                )

                    if st.button("Apply Mapping", type="primary"):
                        mapping = {k: v.strip() for k, v in inputs.items() if v and v.strip()}
                        if not mapping:
                            st.warning("Provide at least one target name before applying.")
                        else:
                            result = apply_manual_speaker_mapping(active_dir, mapping)
                            if result.returncode == 0:
                                st.session_state["speaker_flash"] = {
                                    "status": "success",
                                    "message": result.stdout or "Manual mapping applied.",
                                }
                                st.rerun()
                            else:
                                st.error(result.stderr or "Failed to apply mapping.")
        else:
            st.caption("All segments have assigned speaker labels.")
else:
    st.caption(
        "Speaker identification will populate final transcript artifacts with resolved speaker names."
    )

st.divider()
st.subheader("Cluster Snippets (Manual Review)")
st.caption(
    "Listen to diarization segments, then apply names directly to each clip or via the mapping above."
)
summary_path = (
    _PROJECT_ROOT
    / "exports"
    / "speaker_review"
    / (active_label or active_dir.name)
    / "summary.json"
)
if not summary_path.exists():
    summary_path = _PROJECT_ROOT / "exports" / "speaker_review" / active_dir.name / "summary.json"

segment_controls: list[dict[str, object]] = []

if summary_path.exists():
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Unable to parse review summary.")
    else:
        clusters_payload: dict[str, list[dict[str, object]]] = {}

        if isinstance(data, dict):
            declared_clusters = data.get("clusters")
            if isinstance(declared_clusters, dict):
                for cluster_id, entries in declared_clusters.items():
                    if isinstance(entries, list):
                        clusters_payload[cluster_id] = [
                            entry for entry in entries if isinstance(entry, dict)
                        ]

            legacy_clusters = {
                key: value
                for key, value in data.items()
                if isinstance(value, dict) and "snippets" in value
            }
            for cluster_id, info in legacy_clusters.items():
                snippets = info.get("snippets")
                if isinstance(snippets, list):
                    clusters_payload.setdefault(cluster_id, []).extend(
                        [entry for entry in snippets if isinstance(entry, dict)]
                    )

        if clusters_payload:
            for cluster_id, entries in sorted(clusters_payload.items()):
                valid_entries = [entry for entry in entries if isinstance(entry, dict)]
                if not valid_entries:
                    continue
                with st.expander(f"{cluster_id} ({len(valid_entries)})", expanded=False):
                    for entry in valid_entries[:12]:
                        path = entry.get("audio_path") or entry.get("path")
                        start_val = float(entry.get("start", 0.0) or 0.0)
                        end_val = float(entry.get("end", 0.0) or 0.0)
                        segment_id = entry.get("id")
                        if not isinstance(segment_id, int):
                            segment_id = int(round(start_val * 1000))
                        if isinstance(path, str) and path:
                            audio_player.render(path, label=f"{start_val:.2f}s – {end_val:.2f}s")
                        text = entry.get("text")
                        if isinstance(text, str) and text.strip():
                            st.caption(text.strip())
                        state_key = f"segment_assign::clusters::{active_dir}::{cluster_id}::{segment_id:04d}::{start_val:.3f}"
                        default_value = entry.get("assigned_speaker") or entry.get("speaker") or ""
                        if state_key not in st.session_state:
                            st.session_state[state_key] = str(default_value or "")
                        st.text_input(
                            "Speaker name",
                            key=state_key,
                            help="Blank leaves this clip unchanged.",
                        )
                        if not any(ctrl.get("state_key") == state_key for ctrl in segment_controls):
                            segment_controls.append(
                                {
                                    "state_key": state_key,
                                    "start": start_val,
                                    "end": end_val,
                                }
                            )
                        st.divider()

        segments_block_rendered = False
        if isinstance(data, dict):
            segments = data.get("segments")
            if isinstance(segments, list) and segments:
                segments_by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
                for segment in segments:
                    if isinstance(segment, dict):
                        key = str(segment.get("speaker") or "unknown")
                        segments_by_cluster[key].append(segment)
                if segments_by_cluster:
                    segments_block_rendered = True
                    st.subheader("Segments For Assignment")
                    st.caption(
                        "Each clip stops when a new voice begins. Use these cues to decide speaker names."
                    )
                    for cluster_label, entries in sorted(segments_by_cluster.items()):
                        with st.expander(f"{cluster_label} ({len(entries)})", expanded=False):
                            for entry in entries[:15]:
                                path = entry.get("path") or entry.get("audio_path")
                                start_val = float(entry.get("start", 0.0) or 0.0)
                                end_val = float(entry.get("end", 0.0) or 0.0)
                                segment_id = entry.get("id")
                                if not isinstance(segment_id, int):
                                    segment_id = int(round(start_val * 1000))
                                if isinstance(path, str) and path:
                                    audio_player.render(
                                        path, label=f"{start_val:.2f}s – {end_val:.2f}s"
                                    )
                                text = entry.get("text")
                                if isinstance(text, str) and text.strip():
                                    st.caption(text.strip())
                                state_key = f"segment_assign::segments::{active_dir}::{cluster_label}::{segment_id:04d}::{start_val:.3f}"
                                default_value = (
                                    entry.get("assigned_speaker") or entry.get("speaker") or ""
                                )
                                if state_key not in st.session_state:
                                    st.session_state[state_key] = str(default_value or "")
                                st.text_input(
                                    "Speaker name",
                                    key=state_key,
                                    help="Blank leaves this clip unchanged.",
                                )
                                if not any(
                                    ctrl.get("state_key") == state_key for ctrl in segment_controls
                                ):
                                    segment_controls.append(
                                        {
                                            "state_key": state_key,
                                            "start": start_val,
                                            "end": end_val,
                                        }
                                    )
                                st.divider()

        if not clusters_payload and not segments_block_rendered:
            st.caption("No review snippets listed in summary.json yet.")
else:
    st.caption(
        "No exported review snippets found yet. Use the Diarization page to export segments."
    )

assignments_key = f"segment_controls::{active_dir}"
st.session_state[assignments_key] = segment_controls

if segment_controls:
    if st.button("Apply Segment Assignments", type="primary"):
        pending_controls = st.session_state.get(assignments_key, []) or []
        assignments: list[dict[str, object]] = []
        for entry in pending_controls:
            state_key = entry.get("state_key")
            if not isinstance(state_key, str):
                continue
            value = st.session_state.get(state_key, "")
            speaker_name = value.strip() if isinstance(value, str) else ""
            if not speaker_name:
                continue
            assignments.append(
                {
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "speaker": speaker_name,
                }
            )
        if not assignments:
            st.warning("Provide at least one speaker name before applying assignments.")
        else:
            result = assign_segment_speakers(active_dir, assignments)
            if result.returncode == 0:
                st.session_state["speaker_flash"] = {
                    "status": "success",
                    "message": result.stdout or "Segment assignments applied.",
                }
                st.rerun()
            else:
                st.error(result.stderr or "Failed to apply segment assignments.")
