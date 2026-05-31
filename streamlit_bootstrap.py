"""Helpers for running the Streamlit app from a src-layout repository."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_on_path() -> None:
    """Make the local ``src`` package importable for Streamlit page execution."""

    root = Path(__file__).resolve().parent
    src = root / "src"

    for candidate in (root, src):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
