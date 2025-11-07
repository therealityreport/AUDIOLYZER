"""Global pytest fixtures for Show-Scribe."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory):
    """Provide a temporary directory for sample data assets."""
    return tmp_path_factory.mktemp("show_scribe_data")
