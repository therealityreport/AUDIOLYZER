"""Utilities for configuring Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


def initialise_paths() -> Tuple[Path, Path]:
    """Ensure repository and src directories are importable."""

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "src"

    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))

    return root, src_dir
