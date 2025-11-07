"""Smoke tests ensuring packages import correctly."""

from __future__ import annotations


def test_import_show_scribe_package() -> None:
    import importlib

    assert importlib.import_module("show_scribe") is not None
