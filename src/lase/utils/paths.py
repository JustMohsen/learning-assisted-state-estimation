"""Filesystem locations used by the project.

Raw data is never stored in git. By default it lives in ``<repo>/local_data``
(gitignored except for its README). Set ``LASE_DATA_DIR`` to use another location.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "LASE_DATA_DIR"
DEFAULT_DATA_DIRNAME = "local_data"


def repo_root() -> Path:
    """Repository root (assumes the package is installed from source, as ``uv sync`` does)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Root data directory: ``$LASE_DATA_DIR`` if set, otherwise ``<repo>/local_data``.

    Raises:
        RuntimeError: If the resolved location is not an existing directory.
    """
    value = os.environ.get(DATA_DIR_ENV)
    path = Path(value).expanduser() if value else repo_root() / DEFAULT_DATA_DIRNAME
    if not path.is_dir():
        source = f"{DATA_DIR_ENV}={value}" if value else "default local_data/"
        raise RuntimeError(
            f"Data directory {path} ({source}) does not exist. See local_data/README.md."
        )
    return path


def nclt_dir() -> Path:
    """Root of the NCLT data inside the data directory (may not exist before download)."""
    return data_dir() / "nclt"


def runs_dir() -> Path:
    """Directory for run outputs (gitignored)."""
    return repo_root() / "runs"
