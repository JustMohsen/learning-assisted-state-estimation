"""Filter configuration (tunable, YAML) and derived constant parameters (tensors)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    speed_scale: float | None = None  # None: use the calibration estimate
    gate_prob: float = 0.999


class NonholonomicConfig(Config):
    """Soft non-holonomic pseudo-measurement: axle lateral/vertical body velocity ~ 0.

    Scope stage 3 lists the (learned-trust) lateral no-slip constraint as an extension;
    here it is an optional analytical baseline component (decision pending with the user).
    """

    enabled: bool = False
    sigma_lateral_mps: float = 0.1
    vertical_enabled: bool = False  # body-z velocity ~ 0 is wrong under pitch (v sin(pitch))
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
    min_interval_s: float = 0.0  # skip fixes closer than this to the previous GPS update


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
    fog_handling: FogHandlingConfig = FogHandlingConfig()
    init: InitConfig = InitConfig()
    use_fog: bool = True
    # Off: on training days compensating made heading worse (released FOG data appears to be
    # Earth-rate compensated already); kept as an option.
    earth_rotation: bool = False
    output_every: int = 1  # emit an estimate every n-th propagation step


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
        )
