"""Streamlit helpers for transcript review views."""

from __future__ import annotations

import streamlit as st

from show_scribe.pipelines.transcript.export_text import render_plain_text_with_alignment
from show_scribe.pipelines.transcript.pipeline import TranscriptPipelineResult


def render_alignment_view(result: TranscriptPipelineResult) -> None:
    """Display an alignment-rich transcript for analyst review."""

    st.subheader("Transcript (Alignment View)")
    detailed_text = render_plain_text_with_alignment(result.document)
    st.text(detailed_text)
