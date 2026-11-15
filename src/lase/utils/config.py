"""Validated configuration loading.

Every tunable value lives in a YAML file under ``configs/`` and is parsed into a
pydantic model derived from :class:`Config`. Configs are immutable and reject
unknown keys, so a typo in a YAML file fails loudly instead of being ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base class for all project configs: frozen, unknown keys forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def load_config[C: Config](path: str | Path, model: type[C]) -> C:
    """Load a YAML file and validate it against ``model``.

    Args:
        path: YAML file path.
        model: Config subclass describing the expected structure.

    Returns:
        A validated, immutable config instance.

    Raises:
        ValueError: If the file does not contain a mapping at the top level.
        pydantic.ValidationError: If the contents do not match ``model``.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return model.model_validate(raw)


def dump_config(config: Config, path: str | Path) -> None:
    """Write a config to YAML (used to record the resolved config of each run)."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
