"""All tunable parameters of the project, loaded from ``configs/parameters.yaml``.

``configs/parameters.yaml`` is the single, commented source of every tunable value
(filter, data validity, evaluation, training). Experiments override a subset with small YAML files
that are deep-merged on top (``load_parameters(overrides=[...])``). Measured physical
facts (mounting, origin, FOG sign) stay in ``configs/data/nclt_calibration.yaml``;
simulator scenarios stay in ``configs/sim/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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


class TrainingWindowParams(Config):
    """Chunking for truncated backpropagation (Phase 4, D3)."""

    burn_in_s: float = 20.0
    loss_window_s: float = 10.0
    carry_state: bool = True


class OptimizerParams(Config):
    """Optimiser and learning-rate schedule (Phase 4, D6)."""

    name: Literal["adamw"] = "adamw"
    lr: float = 1.0e-2
    lr_final: float = 1.0e-4
    warmup_steps: int = 20
    schedule: Literal["cosine"] = "cosine"
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


class EarlyStoppingParams(Config):
    """Validation replays and stopping (Phase 4, D6)."""

    max_steps: int = 2000
    val_every_steps: int = 100
    patience: int = 5


class TrainingLossParams(Config):
    """loss = (1 - alpha) * task + alpha * innovation NLL + anchor penalty (Phase 4, D2)."""

    innovation_nll_weight: float = 0.0
    huber_delta: float = 2.0
    pos_scale_m: float = 3.0
    heading_scale_rad: float = 0.03
    vel_scale_mps: float = 0.2
    pos_weight: float = 1.0
    heading_weight: float = 1.0
    vel_weight: float = 1.0
    anchor_penalty_weight: float = 1.0


class AnchorSigmaParams(Config):
    """Prior width of each learned deviation from its anchor (Phase 4, D1)."""

    gnss_sigma: float = 0.7
    wheel_sigma: float = 0.7
    lateral_sigma: float = 0.7
    vertical_sigma: float = 0.7
    acc_noise_density: float = 0.7
    gyr_noise_density: float = 0.7
    fog_angle_random_walk: float = 0.35
    grade_rw: float = 0.7
    wheel_scale_offset: float = 0.01
    encoder_pitch_coef_m: float = 0.04


class SlowGridParams(Config):
    """Full-session grid for the slow parameters (Phase 4, D7)."""

    fog_bias_prior_radps: tuple[float, ...] = (3.0e-5, 1.0e-4, 3.0e-4)
    fog_bias_rw: tuple[float, ...] = (3.0e-7, 1.0e-6, 3.0e-6)
    gnss_error_sigma_m: tuple[float, ...] = (1.5, 2.53)
    gnss_error_tau_s: tuple[float, ...] = (20.0, 55.0)
    round2_fog_bias_rw_factors: tuple[float, ...] = (0.333, 1.0, 3.0)
    round3_enabled: bool = True
    round3_bias_rw_factors: tuple[float, ...] = (0.333, 3.0)


class TrainingParams(Config):
    """Method B training (Phase 4, plans/PHASE_4_TRAINING.md)."""

    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"
    batch_size: int = 64
    seeds_dev: tuple[int, ...] = (0,)
    seeds_report: tuple[int, ...] = (0, 1, 2)
    windows: TrainingWindowParams = TrainingWindowParams()
    optimizer: OptimizerParams = OptimizerParams()
    early_stopping: EarlyStoppingParams = EarlyStoppingParams()
    loss: TrainingLossParams = TrainingLossParams()
    anchor_sigma: AnchorSigmaParams = AnchorSigmaParams()
    slow_grid: SlowGridParams = SlowGridParams()


class Parameters(Config):
    """Every tunable value of the project."""

    filter: FilterConfig = FilterConfig()
    data: DataParams = DataParams()
    evaluation: EvaluationParams = EvaluationParams()
    training: TrainingParams = TrainingParams()


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
