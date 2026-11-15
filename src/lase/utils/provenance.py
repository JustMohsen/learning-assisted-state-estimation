"""Run provenance: every run records what produced it.

A run directory holds the resolved config, the git commit (and whether the
working tree was dirty), the seed, library versions and any extra metadata
such as the data manifest used.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lase.utils.config import Config, dump_config
from lase.utils.paths import repo_root, runs_dir


def git_state() -> dict[str, Any]:
    """Current commit hash and dirty flag, or ``None`` values outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def create_run_dir(
    name: str,
    config: Config,
    seed: int,
    extra: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Create a timestamped run directory and write its provenance.

    Args:
        name: Short run name, e.g. ``"baseline-eskf"``.
        config: Fully resolved config for the run.
        seed: Random seed used.
        extra: Additional metadata (e.g. data manifest, split name).
        base_dir: Parent directory; defaults to ``runs/``.

    Returns:
        Path of the new run directory, containing ``config.yaml`` and ``provenance.json``.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (base_dir or runs_dir()) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_config(config, run_dir / "config.yaml")
    record = {
        "name": name,
        "created_utc": stamp,
        "seed": seed,
        "git": git_state(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "platform": platform.platform(),
        "extra": extra or {},
    }
    (run_dir / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run_dir
