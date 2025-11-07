"""Bleep detection review UI."""

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

import csv
import json
from pathlib import Path

import streamlit as st

from ui.streamlit_app.bootstrap import initialise_paths

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (  # noqa: E402
    get_stage_status,
    read_metadata,
    resolve_artifact_path,
    run_bleep_detection,
    update_metadata,
)
from ui.streamlit_app.components import audio_player  # noqa: E402
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Bleep Review", layout="wide")
active_label, active_dir, _ = render_global_sidebar()
flash = st.session_state.pop("bleep_flash", None)
if flash:
    status = flash.get("status", "info")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)
st.title("Bleep Review")
if not active_dir:
    st.info("Select an episode to review bleep detections.")
    st.stop()
st.caption(f"Active episode: **{active_label or active_dir.name}**")
metadata = read_metadata(active_dir) or {}
prefix = metadata.get("artifact_prefix") or active_dir.name
status = get_stage_status(active_dir)
diarization_ready = status.get("diarization")
if not diarization_ready:
    st.warning(
        "Diarization artifacts not found. Complete diarization before running bleep detection."
    )
trigger = st.button(
    "Detect Bleeps",
    type="primary",
    disabled=not diarization_ready,
)
if trigger:
    result = run_bleep_detection(active_dir)
    if result.returncode == 0:
        st.session_state["bleep_flash"] = {
            "status": "success",
            "message": result.stdout or "Bleep detection completed.",
        }
    else:
        st.session_state["bleep_flash"] = {
            "status": "error",
            "message": result.stderr or "Bleep detection failed.",
        }
    st.rerun()
bleeps_path = resolve_artifact_path(active_dir, "bleeps")
payload: object | None = None
events: list[dict[str, object]] = []
if bleeps_path and bleeps_path.exists():
    render_artifact_badge(bleeps_path, bleeps_path.name)
    try:
        payload = json.loads(bleeps_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Unable to parse bleeps JSON.")
    else:
        if isinstance(payload, list):
            events = [dict(item) for item in payload]
        elif isinstance(payload, dict):
            if isinstance(payload.get("events"), list):
                events = [dict(item) for item in payload["events"]]
        else:
            st.warning("Unexpected bleep payload format.")
if events:
    st.subheader("Detected Bleeps")
    edited_events = st.data_editor(
        events,
        num_rows="dynamic",
        use_container_width=True,
        key="bleep_editor",
    )
    preview_sources = [
        resolve_artifact_path(active_dir, "audio_processed"),
        resolve_artifact_path(active_dir, "audio_extracted"),
    ]
    preview_options = [path for path in preview_sources if path and path.exists()]
    if preview_options:
        preview_audio = st.selectbox(
            "Audio preview source",
            options=preview_options,
            format_func=lambda path: path.name,
        )
        audio_player.render(preview_audio, label="Episode audio preview")
    else:
        st.caption("Audio file missing for preview.")
    save_cols = st.columns(2)

    def _write_payload(path: Path, rows: list[dict[str, object]]) -> None:
        if isinstance(payload, dict) and "events" in payload:
            new_payload = dict(payload)
            new_payload["events"] = rows
            path.write_text(json.dumps(new_payload, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if save_cols[0].button("Save JSON"):
        target_json = (
            bleeps_path or (active_dir / "transcripts" / f"{prefix or active_dir.name}_bleeps.json")
        ).resolve()
        target_json.parent.mkdir(parents=True, exist_ok=True)
        _write_payload(target_json, edited_events)
        try:
            rel_path = str(target_json.relative_to(active_dir))
        except ValueError:
            rel_path = str(target_json)
        update_metadata(active_dir, {"artifacts": {"bleeps": rel_path}})
        st.success(f"Updated {target_json.name}.")
    if save_cols[1].button("Export CSV"):
        csv_base = bleeps_path.parent if bleeps_path else active_dir / "transcripts"
        csv_base.mkdir(parents=True, exist_ok=True)
        csv_name = f"{prefix or active_dir.name}_bleeps.csv"
        csv_path = csv_base / csv_name
        fieldnames = sorted({key for item in edited_events for key in item.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(edited_events)
        st.success(f"Wrote {csv_path.name}.")
else:
    st.caption("Run bleep detection to populate bleep annotations.")
