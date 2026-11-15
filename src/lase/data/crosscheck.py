"""Phase 1 cross-sensor validation checks against the independent reference.

All functions here are *offline* analyses (they interpolate across time and use the
reference). They validate frames, timing, signs and scales; they are never estimator
inputs. Check numbering follows plans/PHASE_1_DATA_FOUNDATION.md, section 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lase.data.calibration import NcltCalibration
from lase.data.conversions import interp_at, rate_on_grid, wheel_forward_speed
from lase.data.raw import F64, I64
from lase.data.reference import Reference
from lase.data.session import SensorStreams
from lase.geometry.euler import euler_zyx_to_matrix


def common_grid(
    streams: SensorStreams, ref: Reference, rate_hz: float, margin_s: float = 2.0
) -> I64:
    """Regular time grid inside the overlap of all inputs and the valid reference."""
    tg = ref.gt.t_us[ref.gt.valid]
    t0 = max(streams.imu.t_us[0], streams.fog.t_us[0], streams.wheels.t_us[0], tg[0])
    t1 = min(streams.imu.t_us[-1], streams.fog.t_us[-1], streams.wheels.t_us[-1], tg[-1])
    step = round(1e6 / rate_hz)
    grid: I64 = np.arange(t0 + int(margin_s * 1e6), t1 - int(margin_s * 1e6), step, dtype=np.int64)
    return grid


def fit_similarity_2d(src: F64, dst: F64) -> tuple[float, float, F64]:
    """Least-squares similarity dst ~ s R src + t (Umeyama). Returns (angle_rad, scale, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / np.mean(np.sum(a**2, axis=1)))
    t: F64 = mu_d - scale * R @ mu_s
    return float(np.arctan2(R[1, 0], R[0, 0])), scale, t


@dataclass(frozen=True)
class GpsCheck:
    """V1: consumer GPS vs reference antenna position (horizontal)."""

    n: int
    median_err_m: float
    p95_err_m: float
    mean_residual_ne_m: tuple[float, float]
    rotation_deg: float
    scale_minus_1: float


def check_gps(streams: SensorStreams, ref: Reference, cal: NcltCalibration) -> GpsCheck:
    """Compare valid GPS fixes with the reference antenna position at the fix times."""
    g = streams.gps
    m = g.valid
    t = g.t_us[m]
    p_b, rpy = ref.pose_at(t)
    ok = np.isfinite(p_b).all(1)
    C_wb = euler_zyx_to_matrix(rpy[ok, 0], rpy[ok, 1], rpy[ok, 2])
    p_ant = p_b[ok] + C_wb @ cal.sensors.gps.l_b
    meas = streams.gnss_ned[m][ok]
    r = meas[:, :2] - p_ant[:, :2]
    err = np.linalg.norm(r, axis=1)
    ang, scale, _ = fit_similarity_2d(p_ant[:, :2], meas[:, :2])
    return GpsCheck(
        n=int(ok.sum()),
        median_err_m=float(np.median(err)),
        p95_err_m=float(np.percentile(err, 95)),
        mean_residual_ne_m=(float(r[:, 0].mean()), float(r[:, 1].mean())),
        rotation_deg=float(np.degrees(ang)),
        scale_minus_1=scale - 1.0,
    )


def xcorr_lag(a: F64, b: F64, max_lag: int) -> tuple[int, float]:
    """Lag (samples) maximising the normalised correlation of ``a[k]`` with ``b[k + lag]``."""
    best = (0, -np.inf)
    a0 = (a - a.mean()) / a.std()
    b0 = (b - b.mean()) / b.std()
    n = len(a0)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.mean(a0[: n - lag] * b0[lag:]))
        else:
            c = float(np.mean(a0[-lag:] * b0[: n + lag]))
        if c > best[1]:
            best = (lag, c)
    return best


@dataclass(frozen=True)
class YawRateCheck:
    """V2: yaw-rate agreement between IMU gyro z, FOG and reference heading derivative."""

    fog_lag_ms: float  # positive: FOG lags the reference
    fog_corr: float
    fog_slope: float  # FOG rate regressed on reference rate
    imu_lag_ms: float
    imu_corr: float
    imu_slope: float


def _slope(x: F64, y: F64) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.sum(x[m] * y[m]) / np.sum(x[m] ** 2))


def yaw_rates(streams: SensorStreams, ref: Reference, grid: I64) -> tuple[F64, F64, F64]:
    """Reference, FOG and IMU-gyro-z yaw rates (rad/s) on ``grid`` (offline)."""
    m = ref.gt.valid
    gt_rate = rate_on_grid(ref.gt.t_us[m], np.unwrap(ref.gt.rpy[m, 2]), grid)
    fog_rate = rate_on_grid(streams.fog.t_us, streams.fog_angle, grid)
    imu_rate = interp_at(streams.imu.t_us, streams.imu.gyr[:, 2], grid)
    return gt_rate, fog_rate, imu_rate


def check_yaw_rate(
    streams: SensorStreams, ref: Reference, rate_hz: float = 100.0, max_lag_s: float = 1.0
) -> YawRateCheck:
    """Cross-correlate yaw rates to find time offsets, signs and scales."""
    grid = common_grid(streams, ref, rate_hz)
    gt_rate, fog_rate, imu_rate = yaw_rates(streams, ref, grid)
    L = int(max_lag_s * rate_hz)
    fl, fc = xcorr_lag(gt_rate, fog_rate, L)
    il, ic = xcorr_lag(gt_rate, imu_rate, L)
    # signed correlation at zero lag reveals sign flips (xcorr maximises positive correlation)
    return YawRateCheck(
        fog_lag_ms=fl * 1e3 / rate_hz,
        fog_corr=fc,
        fog_slope=_slope(gt_rate, fog_rate),
        imu_lag_ms=il * 1e3 / rate_hz,
        imu_corr=ic,
        imu_slope=_slope(gt_rate, imu_rate),
    )


@dataclass(frozen=True)
class WheelSpeedCheck:
    """V3: mean wheel speed vs reference forward speed."""

    n: int
    slope: float  # wheel speed regressed on reference speed
    median_abs_residual_mps: float
    mean_residual_mps: float


def check_wheel_speed(
    streams: SensorStreams, ref: Reference, rate_hz: float = 10.0
) -> WheelSpeedCheck:
    """Compare the wheelbase-centre speed (paper Eq. 4) with the reference forward speed."""
    grid = common_grid(streams, ref, rate_hz)
    w = streams.wheels
    v_wheel = interp_at(w.t_us, wheel_forward_speed(w.left, w.right), grid)
    v_ref, mask = ref.forward_speed(grid)
    m = mask & np.isfinite(v_wheel) & (np.abs(v_ref) > 0.3)
    r = v_wheel[m] - v_ref[m]
    return WheelSpeedCheck(
        n=int(m.sum()),
        slope=_slope(v_ref[m], v_wheel[m]),
        median_abs_residual_mps=float(np.median(np.abs(r))),
        mean_residual_mps=float(np.mean(r)),
    )


@dataclass(frozen=True)
class TrackWidthCheck:
    """V4: differential wheel speed vs FOG yaw rate, giving the effective track width."""

    n: int
    track_width_m: float
    r2: float


def check_track_width(
    streams: SensorStreams, ref: Reference, fog_sign: float = 1.0, rate_hz: float = 10.0
) -> TrackWidthCheck:
    """Fit (left - right) = W * yaw_rate over turning samples, with yaw rate from the FOG."""
    grid = common_grid(streams, ref, rate_hz)
    w = streams.wheels
    diff = interp_at(w.t_us, w.left - w.right, grid)
    rate = fog_sign * rate_on_grid(streams.fog.t_us, streams.fog_angle, grid)
    m = np.isfinite(diff) & np.isfinite(rate) & (np.abs(rate) > 0.05)
    x, y = rate[m], diff[m]
    W = float(np.sum(x * y) / np.sum(x**2))
    ss_res = float(np.sum((y - W * x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return TrackWidthCheck(n=int(m.sum()), track_width_m=W, r2=1 - ss_res / ss_tot)
