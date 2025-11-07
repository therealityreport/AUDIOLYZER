"""Analytics visualisations for processed episodes."""

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

ROOT, SRC_DIR = initialise_paths()
from show_scribe.ui_ops import (
    compute_analytics,
    get_stage_status,
    resolve_artifact_path,
)  # noqa: E402
from ui.streamlit_app.components.artifact_badge import render_artifact_badge  # noqa: E402
from ui.streamlit_app.components.layout import render_global_sidebar  # noqa: E402

st.set_page_config(page_title="Analytics", layout="wide")
active_label, active_dir, _ = render_global_sidebar()
flash = st.session_state.pop("analytics_flash", None)
if flash:
    status = flash.get("status", "info")
    message = flash.get("message", "")
    if status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.success(message)
st.title("Episode Analytics")
if not active_dir:
    st.info("Select an episode to view analytics.")
    st.stop()
st.caption(f"Active episode: **{active_label or active_dir.name}**")
status = get_stage_status(active_dir)
analytics_available = status.get("analytics")
trigger = st.button(
    "Compute Analytics",
    type="primary",
    disabled=False,
)
if trigger:
    result = compute_analytics(active_dir)
    if result.returncode == 0:
        st.session_state["analytics_flash"] = {
            "status": "success",
            "message": result.stdout or "Analytics generated.",
        }
    else:
        st.session_state["analytics_flash"] = {
            "status": "error",
            "message": result.stderr or "Analytics computation failed.",
        }
    st.rerun()
analytics_path = resolve_artifact_path(active_dir, "analytics")
if analytics_path and analytics_path.exists():
    render_artifact_badge(analytics_path, analytics_path.name)
    try:
        payload = json.loads(analytics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Unable to parse analytics JSON.")
    else:
        speaking_time = payload.get("speaking_time") or payload.get("speaking_time_seconds")
        if isinstance(speaking_time, dict):
            st.subheader("Speaking Time by Person")
            st.bar_chart(speaking_time)
        bleep_stats = payload.get("bleeps")
        if isinstance(bleep_stats, list):
            st.subheader("Bleep Events")
            st.table(bleep_stats)
        timeline = payload.get("timeline")
        if isinstance(timeline, list):
            st.subheader("Timeline")
            st.line_chart(
                {
                    event.get("time"): event.get("value", 0)
                    for event in timeline
                    if isinstance(event, dict)
                }
            )
else:
    st.caption("Compute analytics to populate analytics artifacts.")
