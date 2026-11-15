"""Sensor models: generate measurements from the truth trajectory with known errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lase.data.calibration import NcltCalibration
from lase.data.geodesy import LocalProjection
from lase.data.raw import F64, I64, B
from lase.geometry.constants import GRAVITY_MPS2
from lase.sim.config import FogErrors, GpsErrors, ImuErrors, WheelErrors
from lase.sim.trajectory import TruthTrajectory

Span = tuple[int, int]


def random_walk(t_us: I64, init_sigma: float, rw: float, dim: int, rng: np.random.Generator) -> F64:
    """Bias trajectory: constant N(0, init_sigma^2) start plus random walk of intensity ``rw``."""
    b0 = rng.normal(0.0, init_sigma, dim) if init_sigma > 0 else np.zeros(dim)
    out = np.tile(b0, (len(t_us), 1))
    if rw > 0 and len(t_us) > 1:
        dt = np.diff(t_us) * 1e-6
        steps = rng.normal(0.0, 1.0, (len(dt), dim)) * (rw * np.sqrt(dt))[:, None]
        out[1:] += np.cumsum(steps, axis=0)
    return out


def poisson_spans(
    t0_us: int, t1_us: int, rate_per_s: float, dur_s: tuple[float, float], rng: np.random.Generator
) -> list[Span]:
    """Non-overlapping random spans with Poisson arrivals and uniform durations."""
    spans: list[Span] = []
    if rate_per_s <= 0:
        return spans
    t = float(t0_us)
    while True:
        t += rng.exponential(1.0 / rate_per_s) * 1e6
        if t >= t1_us:
            break
        d = rng.uniform(*dur_s) * 1e6
        spans.append((int(t), int(min(t + d, t1_us))))
        t += d
    return spans


def in_spans(t_us: I64, spans: list[Span]) -> B:
    """Mask of times inside any span."""
    m = np.zeros(len(t_us), dtype=bool)
    for a, b in spans:
        m |= (t_us >= a) & (t_us < b)
    return m


@dataclass(frozen=True)
class ImuSim:
    acc: F64
    gyr: F64
    acc_bias: F64
    gyr_bias: F64
    acc_scale: F64
    gyr_scale: F64


def simulate_imu(
    traj: TruthTrajectory,
    t_us: I64,
    cal: NcltCalibration,
    err: ImuErrors,
    rng: np.random.Generator,
    earth_rate_w: F64 | None = None,
) -> ImuSim:
    """Specific force and angular rate at the IMU, in IMU axes, with errors.

    If ``earth_rate_w`` (NED, rad/s) is given, the gyro also senses the Earth rate.
    """
    st = traj.at(t_us)
    lever = cal.sensors.imu.l_b
    C_bs = cal.sensors.imu.C_bs
    w, wd = st.omega_b, st.omega_dot_b
    a_lever_b = np.cross(wd, lever) + np.cross(w, np.cross(w, lever))
    a_imu_w = st.a_w + np.einsum("qij,qj->qi", st.C_wb, a_lever_b)
    g_w = np.array([0.0, 0.0, GRAVITY_MPS2])
    f_b = np.einsum("qji,qj->qi", st.C_wb, a_imu_w - g_w)  # C_wb^T (a - g)
    f_s = f_b @ C_bs  # C_bs^T f_b
    if earth_rate_w is not None:
        w = w + np.einsum("qji,j->qi", st.C_wb, earth_rate_w)  # + C_wb^T w_ie
    w_s = w @ C_bs
    acc_scale = rng.normal(0, err.acc_scale_sigma, 3) if err.acc_scale_sigma > 0 else np.zeros(3)
    gyr_scale = rng.normal(0, err.gyr_scale_sigma, 3) if err.gyr_scale_sigma > 0 else np.zeros(3)
    ba = random_walk(t_us, err.acc_bias_init_mps2, err.acc_bias_rw_mps2_sqrts, 3, rng)
    bg = random_walk(t_us, err.gyr_bias_init_radps, err.gyr_bias_rw_radps_sqrts, 3, rng)
    acc = (1 + acc_scale) * f_s + ba + rng.normal(0, 1, f_s.shape) * err.acc_noise_mps2
    gyr = (1 + gyr_scale) * w_s + bg + rng.normal(0, 1, w_s.shape) * err.gyr_noise_radps
    return ImuSim(acc, gyr, ba, bg, acc_scale, gyr_scale)


@dataclass(frozen=True)
class FogSim:
    angle_released: F64
    bias: F64  # rad/s at each sample


def simulate_fog(
    traj: TruthTrajectory,
    t_us: I64,
    err: FogErrors,
    rng: np.random.Generator,
    earth_rate_w: F64 | None = None,
) -> FogSim:
    """Released FOG angle: sign x (body-z angle + integrated bias + noise), wrapped to [0, 2pi)."""
    s = (t_us - traj.t0_us) * 1e-6
    angle = traj.body_z_angle(earth_rate_w)(s)
    bias = random_walk(t_us, err.bias_init_radps, err.bias_rw_radps_sqrts, 1, rng)[:, 0]
    dt = np.diff(t_us, prepend=t_us[0]) * 1e-6
    angle = angle + np.cumsum(bias * dt) + rng.normal(0, 1, len(t_us)) * err.angle_noise_rad
    released = np.mod(err.released_sign * angle, 2 * np.pi)
    return FogSim(released, bias)


@dataclass(frozen=True)
class WheelSim:
    left: F64
    right: F64
    slip: B
    slip_spans: list[Span]


def simulate_wheels(
    traj: TruthTrajectory, t_us: I64, err: WheelErrors, rng: np.random.Generator
) -> WheelSim:
    """Left/right wheel speeds at the axle centre with scale, noise and slip episodes."""
    st = traj.at(t_us)
    v_b = np.einsum("qji,qj->qi", st.C_wb, st.v_w)
    v, r = v_b[:, 0], st.omega_b[:, 2]
    half = 0.5 * err.track_width_m
    left = err.speed_scale * (v + r * half)  # positive yaw (right turn): left faster
    right = err.speed_scale * (v - r * half)
    spans = poisson_spans(
        int(t_us[0]), int(t_us[-1]), err.slip.rate_per_min / 60.0, err.slip.duration_s, rng
    )
    slip = np.zeros(len(t_us), dtype=bool)
    for a, b in spans:
        m = (t_us >= a) & (t_us < b)
        slip |= m
        k = 1.0 + rng.uniform(*err.slip.overspeed_frac)
        if rng.uniform() < err.slip.asymmetric_prob:
            if rng.uniform() < 0.5:
                left[m] *= k
            else:
                right[m] *= k
        else:
            left[m] *= k
            right[m] *= k
    left = left + rng.normal(0, 1, len(t_us)) * err.noise_mps
    right = right + rng.normal(0, 1, len(t_us)) * err.noise_mps
    return WheelSim(left, right, slip, spans)


def episode_offsets(
    t_us: I64,
    rate_per_hour: float,
    dur_median_s: float,
    dur_log_sigma: float,
    sample_magnitude: object,
    rng: np.random.Generator,
) -> tuple[F64, list[Span]]:
    """Piecewise-constant horizontal offsets (N, 3) during Poisson episodes."""
    off = np.zeros((len(t_us), 3))
    spans: list[Span] = []
    if rate_per_hour <= 0 or len(t_us) == 0:
        return off, spans
    t = float(t_us[0])
    while True:
        t += rng.exponential(3600.0 / rate_per_hour) * 1e6
        if t >= t_us[-1]:
            break
        d = float(np.exp(np.log(dur_median_s) + dur_log_sigma * rng.normal())) * 1e6
        a, b = int(t), int(min(t + d, t_us[-1]))
        mag = float(sample_magnitude())  # type: ignore[operator]
        ang = rng.uniform(0, 2 * np.pi)
        m = (t_us >= a) & (t_us < b)
        off[m] += mag * np.array([np.cos(ang), np.sin(ang), 0.0])
        spans.append((a, b))
        t = b
    return off, spans


@dataclass(frozen=True)
class GpsSim:
    t_us: I64  # after injected outages
    lat: F64
    lon: F64
    alt: F64
    mode: I64
    error_ned: F64  # total injected error (m)
    sigma_mult: F64
    outlier: B
    held: B
    outage_spans: list[Span]
    episode_spans: list[Span]  # multipath and gross episodes (truth)


def _markov_mode(t_us: I64, frac2: float, dwell_s: float, rng: np.random.Generator) -> I64:
    mode = np.full(len(t_us), 3, dtype=np.int64)
    if frac2 <= 0:
        return mode
    state: int = 2 if rng.uniform() < frac2 else 3
    dt = np.diff(t_us, prepend=t_us[0]) * 1e-6
    # leave mode 2 at rate 1/dwell; enter at rate frac2/((1-frac2) dwell)
    for i in range(len(t_us)):
        rate = 1.0 / dwell_s if state == 2 else frac2 / max(1e-9, (1 - frac2) * dwell_s)
        if rng.uniform() < 1 - np.exp(-rate * dt[i]):
            state = int(5 - state)
        mode[i] = state
    return mode


def simulate_gps(
    traj: TruthTrajectory,
    t_us: I64,
    cal: NcltCalibration,
    err: GpsErrors,
    rng: np.random.Generator,
) -> GpsSim:
    """Consumer-GPS fixes of the antenna with correlated error, bursts, outliers, outages."""
    outages = poisson_spans(
        int(t_us[0]), int(t_us[-1]), err.outage_rate_per_hour / 3600.0, err.outage_duration_s, rng
    )
    t = t_us[~in_spans(t_us, outages)]
    st = traj.at(t)
    p_ant = st.p_w + np.einsum("qij,j->qi", st.C_wb, cal.sensors.gps.l_b)
    n = len(t)
    vf = np.array([1.0, 1.0, err.vertical_factor])
    # first-order Gauss-Markov
    gm = np.zeros((n, 3))
    if err.gm_sigma_m > 0 and n:
        dt = np.diff(t, prepend=t[0]) * 1e-6
        phi = np.exp(-dt / err.gm_tau_s)
        q = err.gm_sigma_m * np.sqrt(1 - phi**2)
        gm[0] = rng.normal(0, err.gm_sigma_m, 3)
        z = rng.normal(0, 1, (n, 3))
        for i in range(1, n):
            gm[i] = phi[i] * gm[i - 1] + q[i] * z[i]
    bursts = poisson_spans(
        int(t_us[0]), int(t_us[-1]), err.burst_rate_per_min / 60.0, err.burst_duration_s, rng
    )
    mult = np.where(in_spans(t, bursts), err.burst_sigma_factor, 1.0)
    white = rng.normal(0, 1, (n, 3))
    if err.white_tau_s > 0 and n:  # fast Gauss-Markov (unit variance), then scaled
        dtw = np.diff(t, prepend=t[0]) * 1e-6
        phw = np.exp(-dtw / err.white_tau_s)
        for i in range(1, n):
            white[i] = phw[i] * white[i - 1] + np.sqrt(1 - phw[i] ** 2) * white[i]
    white = white * err.white_sigma_m * mult[:, None]
    outlier = rng.uniform(size=n) < err.outlier_prob
    jumps = rng.normal(0, err.outlier_sigma_m, (n, 3)) * outlier[:, None]
    lo, hi = np.log(err.multipath_offset_m[0]), np.log(err.multipath_offset_m[1])
    mp, mp_spans = episode_offsets(
        t,
        err.multipath_rate_per_hour,
        err.multipath_duration_median_s,
        err.multipath_duration_log_sigma,
        lambda: np.exp(rng.uniform(lo, hi)),
        rng,
    )
    gr, gr_spans = episode_offsets(
        t,
        err.gross_rate_per_hour,
        err.gross_duration_median_s,
        err.gross_duration_log_sigma,
        lambda: min(
            err.gross_offset_max_m,
            err.gross_offset_min_m
            + np.exp(
                np.log(err.gross_offset_excess_median_m) + err.gross_offset_log_sigma * rng.normal()
            ),
        ),
        rng,
    )
    outlier = outlier | (np.linalg.norm(mp + gr, axis=1) > 0)
    mode = _markov_mode(t, err.mode2_frac, err.mode_dwell_s, rng)
    e = (gm + white + jumps) * vf + mp + gr
    e[:, 2] *= np.where(mode == 2, err.mode2_vertical_factor, 1.0)
    meas = p_ant + e
    held = np.zeros(n, dtype=bool)
    if err.held_prob > 0 and n > 1:
        held[1:] = rng.uniform(size=n - 1) < err.held_prob
        for j in np.flatnonzero(held):
            meas[j] = meas[j - 1]
            e[j] = meas[j] - p_ant[j]
    proj = LocalProjection.from_config(cal.local_frame)
    lat, lon, alt = proj.to_geodetic(meas)
    return GpsSim(
        t, lat, lon, alt, mode, e, mult, outlier, held, outages, sorted(mp_spans + gr_spans)
    )
