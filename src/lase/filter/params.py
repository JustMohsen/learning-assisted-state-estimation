"""Filter configuration (tunable, YAML) and derived constant parameters (tensors)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from lase.data.calibration import NcltCalibration
from lase.geometry.constants import GRAVITY_MPS2, earth_rate_ned
from lase.utils.config import Config, load_config


class ProcessNoise(Config):
    """Continuous-time noise densities."""

    acc_noise_density: float = 0.02  # m/s^2/sqrt(Hz) (white specific-force noise, incl. vibration)
    gyr_noise_density: float = 0.002  # rad/s/sqrt(Hz)
    fog_angle_random_walk: float = (
        2e-5  # rad/sqrt(s); KVH DSP-3000 datasheet ARW 0.0667 deg/sqrt(h)
    )
    acc_bias_rw: float = 1e-4  # m/s^2/sqrt(s)
    gyr_bias_rw: float = 1e-5  # rad/s/sqrt(s)
    fog_bias_rw: float = 1e-7  # rad/s/sqrt(s)
    wheel_scale_rw: float = 0.0  # 1/sqrt(s); 0 with a zero prior keeps the wheel scale fixed


class InitialPrior(Config):
    """Initial standard deviations (common initial-pose prior)."""

    pos_m: float = 0.5
    vel_mps: float = 0.3
    att_rad: tuple[float, float, float] = (0.02, 0.02, 0.02)
    acc_bias_mps2: float = 0.1
    gyr_bias_radps: float = 0.01
    fog_bias_radps: float = 1e-5  # KVH DSP-3000 bias stability ~1 deg/h = 4.8e-6 rad/s
    wheel_scale: float = 0.0  # 0: fixed at the calibration value; > 0: estimated as a state


class WheelUpdateConfig(Config):
    """Forward-speed update."""

    enabled: bool = True
    sigma_mps: float = 0.1
    speed_scale: float | None = None  # None: date trend if enabled, else the calibration value
    # Fixed scale from a linear trend in date fitted on training days (tyre wear/pressure):
    # scale = intercept + per_year * (date - ref_date) / 365.25 (user decision, Phase 3 review)
    scale_trend_enabled: bool = False
    scale_trend_ref_date: dt.date = dt.date(2012, 1, 1)
    scale_trend_intercept: float = 1.054
    scale_trend_per_year: float = 0.0
    # Schmidt "consider" state: the scale stays at its fixed value, but its uncertainty
    # (prior.wheel_scale) is carried in the covariance (honest dead reckoning; user decision)
    scale_consider: bool = False
    gate_prob: float = 0.999
    encoder_pitch_coef_m: float = 0.0  # R_w: reported speed includes R_w * body pitch rate


class NonholonomicConfig(Config):
    """Soft non-holonomic pseudo-measurement: axle lateral/vertical body velocity ~ 0.

    Scope stage 3 lists the (learned-trust) lateral no-slip constraint as an extension;
    here it is an optional analytical baseline component (decision pending with the user).
    """

    enabled: bool = False
    sigma_lateral_mps: float = 0.1
    # Vertical part: "off"; "body" = body-z axle velocity ~ 0 (biased by v sin(pitch));
    # "level" = world-vertical axle velocity ~ 0 (pitch-independent; slopes within sigma).
    # "grade": the axle follows a terrain with estimated slope: v_D = -(grad h . v_horizontal)
    vertical_mode: Literal["off", "body", "level", "grade"] = "off"
    grade_prior_sigma: float = 0.05  # initial terrain-gradient sigma per axis (rise/run)
    grade_rw_per_sqrt_m: float = 0.01  # terrain-gradient random walk per sqrt(metre travelled)
    sigma_vertical_mps: float = 0.1
    gate_prob: float = 0.999


class GnssUpdateConfig(Config):
    """Horizontal GPS position update."""

    enabled: bool = True
    sigma_m: float = 3.0
    gate_prob: float = 0.999
    skip_held: bool = True
    recovery_after_s: float = 90.0  # inflate position covariance after this long of rejections (> most gross GPS episodes)
    recovery_pos_sigma_m: float = 30.0
    recovery_min_rejections: int = 5  # consecutive rejections required before recovery
    reacq_enabled: bool = True
    reacq_trigger_s: float = 5.0  # gap without accepted GPS that starts re-acquisition
    reacq_min_fixes: int = 5  # consecutive gated fixes required
    reacq_window_s: float = 10.0  # the fixes must lie within this window
    reacq_consistency_m: float = 5.0  # max deviation of each innovation from their mean
    reacq_max_offset_m: float = 10.0  # allowed mean innovation at a zero-length gap
    reacq_offset_growth_mps: float = 0.1  # plus this per second of gap (dead-reckoning drift)
    min_interval_s: float = 0.0  # skip fixes closer than this to the previous GPS update
    error_state: bool = False  # model time-correlated GPS error as a Gauss-Markov state
    error_gm_sigma_m: float = 2.53  # stationary sigma of the correlated error (Phase 2 fit)
    error_gm_tau_s: float = 55.0  # correlation time (Phase 2 fit)
    error_white_sigma_m: float = 1.44  # remaining white noise when error_state is on (Phase 2 fit)


class WheelScaleChainConfig(Config):
    """Carry the wheel-scale estimate from one session to the next (by date).

    Session 1 starts at ``first_value`` with ``first_sigma``; later sessions start at the
    previous session's final estimate with sigma = sqrt(final^2 + floor^2 + (growth * days)^2).
    """

    enabled: bool = False
    first_value: float = 1.0
    first_sigma: float = 0.05
    sigma_floor: float = 0.005
    sigma_growth_per_day: float = 0.0005


class FogHandlingConfig(Config):
    """Causal FOG angle interpolation/extrapolation at interval boundaries."""

    max_extrapolation_s: float = 0.05  # beyond this since the last sample: no FOG for the interval
    extrapolation_baseline_s: float = 0.02  # min time span for the extrapolation rate


class InitConfig(Config):
    """Run start."""

    start_offset_s: float = 1.0  # start this long after the first valid reference pose


class FilterConfig(Config):
    """Complete analytical-baseline configuration."""

    name: str = "eskf-baseline"
    noise: ProcessNoise = ProcessNoise()
    prior: InitialPrior = InitialPrior()
    wheel: WheelUpdateConfig = WheelUpdateConfig()
    nonholonomic: NonholonomicConfig = NonholonomicConfig()
    gnss: GnssUpdateConfig = GnssUpdateConfig()
    wheel_scale_chain: WheelScaleChainConfig = WheelScaleChainConfig()
    fog_handling: FogHandlingConfig = FogHandlingConfig()
    init: InitConfig = InitConfig()
    use_fog: bool = True
    # Remove the Earth rate (~10.1 deg/h on the local vertical at NCLT) sensed by gyro and FOG;
    # the per-session FOG/gyro offsets are left to the bias states (user decision, Phase 3 review).
    earth_rotation: bool = True
    output_every: int = 1  # emit an estimate every n-th propagation step

    def for_date(self, date: dt.date) -> FilterConfig:
        """This configuration with the wheel speed scale resolved for ``date``.

        An explicit ``wheel.speed_scale`` wins; otherwise the date trend, if enabled.
        """
        w = self.wheel
        if w.speed_scale is not None or not w.scale_trend_enabled:
            return self
        years = (date - w.scale_trend_ref_date).days / 365.25
        scale = w.scale_trend_intercept + w.scale_trend_per_year * years
        return self.model_copy(update={"wheel": w.model_copy(update={"speed_scale": scale})})


def load_filter_config(name_or_path: str | Path = "baseline") -> FilterConfig:
    """Filter configuration from ``configs/parameters.yaml``.

    ``"baseline"`` returns the parameters file as is; another bare name applies the
    override file ``configs/experiments/<name>.yaml``; a path to a YAML file containing a
    complete FilterConfig is loaded directly.
    """
    from lase.parameters import load_parameters

    p = Path(name_or_path)
    if p.suffix:
        return load_config(p, FilterConfig)
    if str(name_or_path) == "baseline":
        return load_parameters().filter
    return load_parameters(overrides=[str(name_or_path)]).filter


@dataclass(frozen=True)
class FilterParams:
    """Constant tensors derived from calibration + config (float64)."""

    l_imu: Tensor  # (3,) body origin -> IMU, body axes
    l_gps: Tensor  # (3,) body origin -> GPS antenna
    C_b_imu: Tensor  # (3, 3) IMU axes -> body
    fog_sign: float
    wheel_scale: float
    gravity_w: Tensor  # (3,)
    earth_rate_w: Tensor  # (3,) Earth rotation in NED (zeros if not compensated)
    gps_gm_tau_s: float = 55.0  # GPS-error correlation time
    gps_gm_var: float = 0.0  # GPS-error stationary variance (0 = state disabled)
    grade_rw_per_sqrt_m: float = 0.0  # terrain-gradient random walk (0 = state disabled)
    consider_mask: Tensor | None = None  # (N_ERR,) 1 = consider state (never updated)

    @classmethod
    def build(cls, cal: NcltCalibration, cfg: FilterConfig) -> FilterParams:
        """Derive tensors from the calibration config."""

        def t(x: object) -> Tensor:
            return torch.as_tensor(x, dtype=torch.float64)

        scale = cfg.wheel.speed_scale or cal.wheels.speed_scale or 1.0
        return cls(
            l_imu=t(cal.sensors.imu.l_b),
            l_gps=t(cal.sensors.gps.l_b),
            C_b_imu=t(cal.sensors.imu.C_bs),
            fog_sign=cal.fog_convention.angle_sign_to_ned_yaw,
            wheel_scale=float(scale),
            gravity_w=t([0.0, 0.0, GRAVITY_MPS2]),
            earth_rate_w=t(
                earth_rate_ned(cal.local_frame.lat_origin_deg)
                if cfg.earth_rotation
                else (0.0, 0.0, 0.0)
            ),
            gps_gm_tau_s=cfg.gnss.error_gm_tau_s,
            gps_gm_var=cfg.gnss.error_gm_sigma_m**2 if cfg.gnss.error_state else 0.0,
            grade_rw_per_sqrt_m=cfg.nonholonomic.grade_rw_per_sqrt_m
            if cfg.nonholonomic.vertical_mode == "grade"
            else 0.0,
            consider_mask=_consider_mask(cfg),
        )


def _consider_mask(cfg: FilterConfig) -> Tensor | None:
    from lase.filter.state import N_ERR, WS

    if not cfg.wheel.scale_consider:
        return None
    m = torch.zeros(N_ERR, dtype=torch.float64)
    m[WS] = 1.0
    return m
