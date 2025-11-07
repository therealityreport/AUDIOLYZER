"""Validated name selector component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import get_close_matches

import streamlit as st

__all__ = ["render"]


def render(
    *,
    canonical_names: Sequence[str],
    label: str = "Select Cast Member",
    key: str = "name_selector",
    placeholder_label: str = "-- Select --",
    allow_other: bool = True,
    other_label: str = "Other",
    suggestion_limit: int = 3,
    suggestion_cutoff: float = 0.65,
) -> Mapping[str, object]:
    """Render a searchable dropdown with optional fuzzy suggestions."""
    canonical = sorted({name for name in canonical_names if name}, key=str.casefold)

    search_value = st.text_input(
        "Search names",
        key=f"{key}_search",
        placeholder="Start typing to filter...",
    )
    filtered = [
        name for name in canonical if not search_value or search_value.lower() in name.lower()
    ]

    options = [placeholder_label, *filtered]
    if allow_other:
        options.append(other_label)

    selection = st.selectbox(
        label,
        options,
        key=f"{key}_select",
    )

    result: dict[str, object] = {
        "selected": None,
        "is_other": False,
        "suggestions": [],
        "search": search_value,
    }

    if selection == placeholder_label:
        st.help("Choose a canonical name or pick Other to add a custom entry.")
        return result

    if allow_other and selection == other_label:
        other_value = st.text_input(
            "Enter custom name",
            key=f"{key}_other",
            placeholder="Type the speaker name",
        )
        result["is_other"] = True
        result["selected"] = other_value

        if other_value:
            suggestions = get_close_matches(
                other_value,
                canonical,
                n=suggestion_limit,
                cutoff=suggestion_cutoff,
            )
            if suggestions:
                st.caption("Did you mean: " + ", ".join(suggestions))
            result["suggestions"] = suggestions
        return result

    result["selected"] = selection
    st.success(f"Selected canonical name: {selection}")
    return result
