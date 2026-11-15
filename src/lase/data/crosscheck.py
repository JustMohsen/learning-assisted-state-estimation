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
    """V1: consumer GPS vs reference antenna position (horizontal).

    Consumer GPS has large multipath outliers, so the frame check (rotation, scale) is
    fitted on inliers only and residual offsets are reported as medians.
    """

    n: int
    median_err_m: float
    p95_err_m: float
    inlier_frac: float  # error below max(10 m, 3 x median)
    median_residual_ne_m: tuple[float, float]
    rotation_deg: float  # inlier similarity fit, GPS vs reference
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
    med = float(np.median(err))
    inl = err < max(10.0, 3.0 * med)
    ang, scale, _ = fit_similarity_2d(p_ant[inl, :2], meas[inl, :2])
    return GpsCheck(
        n=int(ok.sum()),
        median_err_m=med,
        p95_err_m=float(np.percentile(err, 95)),
        inlier_frac=float(inl.mean()),
        median_residual_ne_m=(float(np.median(r[:, 0])), float(np.median(r[:, 1]))),
        rotation_deg=float(np.degrees(ang)),
        scale_minus_1=scale - 1.0,
    )


def xcorr_lag(a: F64, b: F64, max_lag: int) -> tuple[float, float]:
    """Lag (samples) maximising ``|corr(a[k], b[k + lag])|``, and the signed correlation there.

    The lag is refined below one sample by fitting a parabola through the peak.
    """
    a0 = (a - a.mean()) / a.std()
    b0 = (b - b.mean()) / b.std()
    n = len(a0)
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.empty(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            c[i] = np.mean(a0[: n - lag] * b0[lag:])
        else:
            c[i] = np.mean(a0[-lag:] * b0[: n + lag])
    k = int(np.argmax(np.abs(c)))
    frac = 0.0
    if 0 < k < len(c) - 1:
        y0, y1, y2 = np.abs(c[k - 1 : k + 2])
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            frac = float(0.5 * (y0 - y2) / denom)
    return float(lags[k]) + frac, float(c[k])


def _slope(x: F64, y: F64) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.sum(x[m] * y[m]) / np.sum(x[m] ** 2))


def _cumulative_angle(t_us: I64, rate: F64) -> F64:
    """Trapezoidal integral of a rate signal (rad), starting at 0."""
    dt = np.diff(t_us) * 1e-6
    out = np.zeros(len(t_us))
    out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)
    return out


@dataclass(frozen=True)
class YawRateCheck:
    """V2: yaw agreement between IMU gyro z, FOG and reference heading.

    Lags come from cross-correlating 100 Hz rates (positive: sensor lags the reference).
    Scales are slopes of 1-s heading increments (sensor on reference), which avoids the
    attenuation that differentiation noise causes in rate regressions. A negative slope
    means the sensor's sign is opposite to NED yaw (positive = clockwise from above).
    """

    fog_lag_ms: float
    fog_corr: float
    fog_scale: float
    imu_lag_ms: float
    imu_corr: float
    imu_scale: float


def heading_increments(
    streams: SensorStreams, ref: Reference, grid: I64, window_us: int = 1_000_000
) -> tuple[F64, F64, F64]:
    """Reference, FOG and IMU-gyro-z heading changes over ``[t - w/2, t + w/2]`` (offline)."""
    m = ref.gt.valid
    h = window_us // 2
    gt_yaw = np.unwrap(ref.gt.rpy[m, 2])
    imu_ang = _cumulative_angle(streams.imu.t_us, streams.imu.gyr[:, 2])

    def inc(t: I64, a: F64) -> F64:
        out: F64 = interp_at(t, a, grid + h) - interp_at(t, a, grid - h)
        return out

    return (
        inc(ref.gt.t_us[m], gt_yaw),
        inc(streams.fog.t_us, streams.fog_angle),
        inc(streams.imu.t_us, imu_ang),
    )


def check_yaw_rate(
    streams: SensorStreams, ref: Reference, rate_hz: float = 100.0, max_lag_s: float = 1.0
) -> YawRateCheck:
    """Find time offsets (rate cross-correlation), signs and scales (1-s increments)."""
    grid = common_grid(streams, ref, rate_hz)
    m = ref.gt.valid
    gt_rate = rate_on_grid(ref.gt.t_us[m], np.unwrap(ref.gt.rpy[m, 2]), grid)
    fog_rate = rate_on_grid(streams.fog.t_us, streams.fog_angle, grid)
    imu_rate = interp_at(streams.imu.t_us, streams.imu.gyr[:, 2], grid)
    L = int(max_lag_s * rate_hz)
    fl, fc = xcorr_lag(gt_rate, fog_rate, L)
    il, ic = xcorr_lag(gt_rate, imu_rate, L)
    coarse = common_grid(streams, ref, 10.0)
    d_gt, d_fog, d_imu = heading_increments(streams, ref, coarse)
    return YawRateCheck(
        fog_lag_ms=fl * 1e3 / rate_hz,
        fog_corr=fc,
        fog_scale=_slope(d_gt, d_fog),
        imu_lag_ms=il * 1e3 / rate_hz,
        imu_corr=ic,
        imu_scale=_slope(d_gt, d_imu),
    )


@dataclass(frozen=True)
class WheelSpeedCheck:
    """V3: mean wheel speed vs reference forward speed, both averaged over 1-s windows."""

    n: int
    scale: float  # wheel speed regressed on reference speed
    median_abs_residual_mps: float
    mean_residual_mps: float


def _window_mean(t_us: I64, x: F64, grid: I64, window_us: int) -> F64:
    """*Offline* mean of ``x`` over ``[t - w/2, t + w/2]`` via its time integral."""
    integral = _cumulative_angle(t_us, x)
    h = window_us // 2
    out: F64 = (interp_at(t_us, integral, grid + h) - interp_at(t_us, integral, grid - h)) / (
        window_us * 1e-6
    )
    return out


def check_wheel_speed(
    streams: SensorStreams,
    ref: Reference,
    rate_hz: float = 10.0,
    window_us: int = 1_000_000,
    min_speed_mps: float = 0.3,
) -> WheelSpeedCheck:
    """Compare the wheelbase-centre speed (paper Eq. 4) with the reference forward speed."""
    grid = common_grid(streams, ref, rate_hz)
    w = streams.wheels
    v_wheel = _window_mean(w.t_us, wheel_forward_speed(w.left, w.right), grid, window_us)
    v_ref, mask = ref.forward_speed(grid, half_window_us=window_us // 2)
    m = mask & np.isfinite(v_wheel) & (np.abs(v_ref) > min_speed_mps)
    r = v_wheel[m] - v_ref[m]
    return WheelSpeedCheck(
        n=int(m.sum()),
        scale=_slope(v_ref[m], v_wheel[m]),
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
    streams: SensorStreams,
    ref: Reference,
    fog_sign: float = 1.0,
    rate_hz: float = 10.0,
    window_us: int = 1_000_000,
    min_rate_radps: float = 0.05,
) -> TrackWidthCheck:
    """Fit ``(left - right) = W * yaw_rate`` (1-s window means), with yaw rate from the FOG.

    ``fog_sign`` converts the released FOG angle to NED yaw (positive clockwise).
    """
    grid = common_grid(streams, ref, rate_hz)
    w = streams.wheels
    diff = _window_mean(w.t_us, w.left - w.right, grid, window_us)
    h = window_us // 2
    fog = streams.fog
    rate = (
        fog_sign
        * (
            interp_at(fog.t_us, streams.fog_angle, grid + h)
            - interp_at(fog.t_us, streams.fog_angle, grid - h)
        )
        / (window_us * 1e-6)
    )
    m = np.isfinite(diff) & np.isfinite(rate) & (np.abs(rate) > min_rate_radps)
    x, y = rate[m], diff[m]
    W = float(np.sum(x * y) / np.sum(x**2))
    ss_res = float(np.sum((y - W * x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return TrackWidthCheck(n=int(m.sum()), track_width_m=W, r2=1 - ss_res / ss_tot)
