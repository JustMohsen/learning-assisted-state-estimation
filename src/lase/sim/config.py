"""Simulation configuration: error models and scenarios.

Default magnitudes are *plausible*, not identified: they follow Phase 1 measurements
(rates, wheel scale, track width, GPS error level) and datasheet-order sensor noise.
Every error source can be disabled by setting its magnitude to zero.
"""

from __future__ import annotations

from pathlib import Path

from lase.sim.trajectory import TrajectoryConfig
from lase.utils.config import Config, load_config
from lase.utils.paths import repo_root


class ImuErrors(Config):
    """Microstrain IMU error model (per-sample white noise; bias = initial + random walk)."""

    acc_noise_mps2: float = 0.02
    gyr_noise_radps: float = 0.003
    acc_bias_init_mps2: float = 0.05  # 1-sigma of the constant initial bias, per axis
    gyr_bias_init_radps: float = 0.003
    acc_bias_rw_mps2_sqrts: float = 1e-4  # random-walk intensity
    gyr_bias_rw_radps_sqrts: float = 1e-5
    acc_scale_sigma: float = 0.0  # 1-sigma of constant scale-factor error, per axis
    gyr_scale_sigma: float = 0.0


class FogErrors(Config):
    """KVH FOG error model on the integrated angle."""

    angle_noise_rad: float = 1e-5  # per-sample white noise on the angle
    bias_init_radps: float = 5e-6
    bias_rw_radps_sqrts: float = 1e-7
    released_sign: float = -1.0  # released angle = sign x body-z angle (Phase 1, V2)


class SlipConfig(Config):
    """Wheel-slip episodes (Poisson arrivals)."""

    rate_per_min: float = 0.0
    duration_s: tuple[float, float] = (0.5, 3.0)
    overspeed_frac: tuple[float, float] = (0.1, 0.5)  # wheel speed excess while slipping
    asymmetric_prob: float = 0.5  # probability that only one wheel slips


class WheelErrors(Config):
    """Wheel-speed model."""

    speed_scale: float = 1.054  # Phase 1 training median
    track_width_m: float = 0.560  # Phase 1 estimate
    noise_mps: float = 0.02
    slip: SlipConfig = SlipConfig()


class GpsErrors(Config):
    """Consumer-GPS error model (horizontal metres; vertical scaled)."""

    white_sigma_m: float = 1.5
    gm_sigma_m: float = 2.5  # first-order Gauss-Markov (correlated) error, per axis
    gm_tau_s: float = 60.0
    vertical_factor: float = 1.5
    burst_rate_per_min: float = 0.0  # noise bursts (sigma multiplied)
    burst_duration_s: tuple[float, float] = (10.0, 60.0)
    burst_sigma_factor: float = 4.0
    outlier_prob: float = 0.0
    outlier_sigma_m: float = 20.0
    outage_rate_per_hour: float = 0.0  # injected outages, on top of real gaps
    outage_duration_s: tuple[float, float] = (30.0, 120.0)
    held_prob: float = 0.0
    mode2_frac: float = 0.4  # fraction of time in 2D-fix mode (Markov chain)
    mode_dwell_s: float = 30.0
    mode2_vertical_factor: float = 5.0


class SimConfig(Config):
    """Complete simulation scenario."""

    name: str
    seed: int = 0
    trajectory: TrajectoryConfig = TrajectoryConfig()
    imu: ImuErrors = ImuErrors()
    fog: FogErrors = FogErrors()
    wheels: WheelErrors = WheelErrors()
    gps: GpsErrors = GpsErrors()
    use_real_gps_gaps: bool = True  # keep the session's natural GPS outages


def load_scenario(name_or_path: str | Path) -> SimConfig:
    """Load ``configs/sim/<name>.yaml`` (or an explicit path)."""
    p = Path(name_or_path)
    if not p.suffix:
        p = repo_root() / "configs" / "sim" / f"{name_or_path}.yaml"
    return load_config(p, SimConfig)
