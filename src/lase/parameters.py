"""All tunable parameters of the project, loaded from ``configs/parameters.yaml``.

``configs/parameters.yaml`` is the single, commented source of every tunable value
(filter, data validity, evaluation). Experiments override a subset with small YAML files
that are deep-merged on top (``load_parameters(overrides=[...])``). Measured physical
facts (mounting, origin, FOG sign) stay in ``configs/data/nclt_calibration.yaml``;
simulator scenarios stay in ``configs/sim/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lase.data.raw import RawLimits
from lase.filter.params import FilterConfig
from lase.utils.config import Config
from lase.utils.paths import repo_root

DEFAULT_PARAMETERS_PATH = Path("configs/parameters.yaml")


class DataParams(Config):
    """Data validity and reference handling."""

    limits: RawLimits = RawLimits()
    node_match_tolerance_s: float = 0.05
    velocity_half_window_s: float = 0.05
    max_reference_speed_mps: float = 5.0


class OutageParams(Config):
    """Injected and natural GPS-outage evaluation."""

    durations_s: tuple[float, ...] = (30.0, 60.0, 120.0)
    per_duration: int = 3
    min_separation_s: float = 180.0
    max_natural_gap_s: float = 2.0
    earliest_start_after_init_s: float = 60.0
    natural_min_gap_s: float = 5.0
    curve_offsets_s: tuple[float, ...] = (0, 5, 10, 20, 30, 45, 60, 90, 120)
    curve_sample_tolerance_s: float = 3.0


class EvaluationParams(Config):
    """Scoring."""

    max_staleness_s: float = 0.2
    dense_rate_hz: float = 10.0
    score_start_after_init_s: float = 10.0
    rpe_deltas_s: tuple[float, ...] = (10.0, 60.0)
    outages: OutageParams = OutageParams()


class Parameters(Config):
    """Every tunable value of the project."""

    filter: FilterConfig = FilterConfig()
    data: DataParams = DataParams()
    evaluation: EvaluationParams = EvaluationParams()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return data


def load_parameters(
    path: str | Path | None = None, overrides: list[str | Path] | None = None
) -> Parameters:
    """Load ``configs/parameters.yaml`` (or ``path``) with optional override files merged on top.

    An override given as a bare name refers to ``configs/experiments/<name>.yaml``.
    """
    data = _read(Path(path) if path else repo_root() / DEFAULT_PARAMETERS_PATH)
    for ov in overrides or []:
        p = Path(ov)
        if not p.suffix:
            p = repo_root() / "configs" / "experiments" / f"{ov}.yaml"
        data = deep_merge(data, _read(p))
    return Parameters.model_validate(data)
