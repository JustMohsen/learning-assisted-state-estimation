"""Filesystem locations used by the project.

Raw data is never stored in the repository. Its location is given by the
``LASE_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "LASE_DATA_DIR"


def repo_root() -> Path:
    """Repository root (assumes the package is installed from source, as ``uv sync`` does)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Root directory of the raw NCLT data, from ``LASE_DATA_DIR``.

    Raises:
        RuntimeError: If the variable is unset or does not point to a directory.
    """
    value = os.environ.get(DATA_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"{DATA_DIR_ENV} is not set. Point it at the directory holding the NCLT data "
            "(see .env.example)."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{DATA_DIR_ENV}={value} is not an existing directory.")
    return path


def runs_dir() -> Path:
    """Directory for run outputs (gitignored)."""
    return repo_root() / "runs"
