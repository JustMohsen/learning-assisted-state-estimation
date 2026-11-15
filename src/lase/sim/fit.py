"""Fit simulator error models to real data (offline; uses the reference).

Statistics are computed per session and pooled over *training* days only, so the
fitted simulator does not encode validation/test information. Residuals against the
NCLT reference include reference error, so fitted noise levels are upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lase.data.audit import spans_where
from lase.data.calibration import NcltCalibration
from lase.data.conversions import interp_at
from lase.data.raw import F64, I64, RowFlag
from lase.data.reference import Reference
from lase.data.session import SensorStreams
from lase.geometry.euler import euler_zyx_to_matrix


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [a, b] (inclusive) of consecutive True values."""
    m = np.asarray(mask, dtype=np.int8)
    e = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1, strict=True))


@dataclass
class GpsStats:
    """GPS residual statistics for one or more sessions."""

    hours: float = 0.0
    n_fixes: int = 0
    inlier_var_ne: list[float] = field(default_factory=list)  # per-session pooled-axis variance
    acf_num: F64 = field(default_factory=lambda: np.zeros(301))
    acf_den: F64 = field(default_factory=lambda: np.zeros(301))
    moderate: list[float] = field(default_factory=list)  # residual norms in (10, 50] m
    episodes: list[tuple[float, float]] = field(
        default_factory=list
    )  # (duration s, median offset m)
    mode2_fixes: int = 0
    mode2_runs_s: list[float] = field(default_factory=list)
    mode3_runs_s: list[float] = field(default_factory=list)
    held: int = 0
    var_by_mode: dict[int, list[float]] = field(default_factory=lambda: {2: [], 3: []})


def gps_residuals(
    streams: SensorStreams, ref: Reference, cal: NcltCalibration
) -> tuple[I64, F64, np.ndarray]:
    """Valid-fix times, horizontal residuals (N, 2) vs reference antenna position, and modes."""
    g = streams.gps
    m = g.valid
    t = g.t_us[m]
    p_b, rpy = ref.pose_at(t)
    ok = np.isfinite(p_b).all(1)
    C = euler_zyx_to_matrix(rpy[ok, 0], rpy[ok, 1], rpy[ok, 2])
    ant = p_b[ok] + C @ cal.sensors.gps.l_b
    r = streams.gnss_ned[m][ok, :2] - ant[:, :2]
    return t[ok], r, g.mode[m][ok]


def accumulate_gps(
    stats: GpsStats, streams: SensorStreams, ref: Reference, cal: NcltCalibration
) -> None:
    """Add one session's GPS residual statistics to ``stats``."""
    t, r, mode = gps_residuals(streams, ref, cal)
    if len(t) < 10:
        return
    stats.hours += (t[-1] - t[0]) * 1e-6 / 3600
    stats.n_fixes += len(t)
    held = (streams.gps.flags[streams.gps.valid] & np.uint16(RowFlag.GPS_HELD)) != 0
    stats.held += int(held.sum())
    n = np.linalg.norm(r, axis=1)
    inl = n <= 10.0
    stats.moderate.extend(n[(n > 10) & (n <= 50)].tolist())
    # gross-outlier episodes: fixes > 50 m off, merged across gaps < 5 s
    gross = n > 50.0
    for a, b in runs(gross):
        stats.episodes.append(((t[b] - t[a]) * 1e-6, float(np.median(n[a : b + 1]))))
    # merge episodes separated by short gaps is approximated by run detection on fixes
    ri = r[inl] - np.median(r[inl], axis=0)  # remove the session's median offset
    stats.inlier_var_ne.append(float(ri.var()))
    for md in (2, 3):
        sel = inl & (mode == md)
        if sel.sum() > 100:
            stats.var_by_mode[md].append(float((r[sel] - np.median(r[sel], axis=0)).var()))
    stats.mode2_fixes += int((mode == 2).sum())
    for md, store in ((2, stats.mode2_runs_s), (3, stats.mode3_runs_s)):
        for a, b in runs(mode == md):
            store.append((t[b] - t[a]) * 1e-6)
    # autocorrelation of inlier residuals at 1 Hz within contiguous segments (gaps <= 2 s)
    ti, rr = t[inl], ri
    breaks = np.flatnonzero(np.diff(ti) > 2_000_000)
    for seg in np.split(np.arange(len(ti)), breaks + 1):
        if len(seg) < 50:
            continue
        g1 = np.arange(ti[seg[0]], ti[seg[-1]], 1_000_000, dtype=np.int64)
        x = interp_at(ti[seg], rr[seg], g1)
        L = min(300, len(g1) - 1)
        for k in range(L + 1):
            prod = (x[: len(x) - k] * x[k:]).sum(1)  # sum over N and E
            stats.acf_num[k] += prod.sum()
            stats.acf_den[k] += 2 * len(prod)


def fit_gauss_markov(acf_num: F64, acf_den: F64, max_lag: int = 300) -> tuple[float, float, float]:
    """Fit ACF(k) = var_w [k=0] + var_gm exp(-k/tau) to pooled 1-Hz autocovariances.

    Returns (white sigma, Gauss-Markov sigma, tau seconds).
    """
    from scipy.optimize import curve_fit

    cov = acf_num[: max_lag + 1] / np.maximum(acf_den[: max_lag + 1], 1)
    k = np.arange(1, max_lag + 1, dtype=np.float64)

    def model(kk: F64, var_gm: float, tau: float) -> F64:
        out: F64 = var_gm * np.exp(-kk / tau)
        return out

    (var_gm, tau), _ = curve_fit(
        model, k, cov[1:], p0=[cov[1], 60.0], bounds=([0, 1], [np.inf, 5000])
    )
    var_w = max(cov[0] - var_gm, 0.0)
    return float(np.sqrt(var_w)), float(np.sqrt(var_gm)), float(tau)


@dataclass
class WheelStats:
    """Wheel-speed residuals vs scaled reference forward speed."""

    moving_min: float = 0.0
    resid: list[F64] = field(default_factory=list)
    events: list[tuple[float, float]] = field(
        default_factory=list
    )  # (duration s, mean |resid| / speed)


def accumulate_wheels(
    stats: WheelStats, streams: SensorStreams, ref: Reference, scale: float
) -> None:
    """Per-sample residual of the mean wheel speed vs scale x reference forward speed."""
    w = streams.wheels
    t = w.t_us
    v_ref, ok = ref.forward_speed(t, half_window_us=50_000)
    z = 0.5 * (w.left + w.right)
    res = z - scale * v_ref
    moving = ok & (np.abs(v_ref) > 0.3)
    if moving.sum() < 100:
        return
    stats.moving_min += float(moving.sum() / (len(t) / ((t[-1] - t[0]) * 1e-6)) / 60)
    stats.resid.append(res[moving])
    # slip-like excursions: |residual| (smoothed ~0.25 s) > 0.3 m/s for >= 0.5 s while moving
    k = 10
    sm = np.convolve(np.where(moving, res, 0.0), np.ones(k) / k, mode="same")
    for a, b in spans_where(t, moving & (np.abs(sm) > 0.3), 0.5):
        sel = (t >= a) & (t <= b)
        rel = float(np.mean(np.abs(res[sel]) / np.maximum(np.abs(v_ref[sel]), 0.3)))
        stats.events.append(((b - a) * 1e-6, rel))


def at_rest_imu_std(streams: SensorStreams, min_s: float = 5.0) -> tuple[F64, F64, int]:
    """Median per-span standard deviation of accelerometer and gyro during stationary spans."""
    w = streams.wheels
    v = 0.5 * (w.left + w.right)
    k = max(1, round(1e6 / np.median(np.diff(w.t_us))))
    sm = np.convolve(v, np.ones(k) / k, mode="same")
    spans = spans_where(w.t_us, np.abs(sm) < 0.05, min_s)
    imu = streams.imu
    sa, sg = [], []
    for a, b in spans:
        sel = (imu.t_us >= a) & (imu.t_us <= b)
        if sel.sum() > 50:
            sa.append(imu.acc[sel].std(0))
            sg.append(imu.gyr[sel].std(0))
    if not sa:
        return np.full(3, np.nan), np.full(3, np.nan), 0
    return np.median(np.array(sa), 0), np.median(np.array(sg), 0), len(sa)


def _hp_std(x: F64, k: int = 470) -> F64:
    """Std of a signal after removing a ~10 s moving mean (per axis)."""
    sm = np.stack([np.convolve(x[:, i], np.ones(k) / k, mode="same") for i in range(x.shape[1])], 1)
    out: F64 = (x - sm)[k:-k].std(0)
    return out


def session_stats(
    streams: SensorStreams, ref: Reference, cal: NcltCalibration, scale: float
) -> dict[str, float]:
    """Key error statistics of one session (real or simulated), for realism comparison."""
    gs, ws = GpsStats(), WheelStats()
    accumulate_gps(gs, streams, ref, cal)
    accumulate_wheels(ws, streams, ref, scale)
    _, r, _ = gps_residuals(streams, ref, cal)
    n = np.linalg.norm(r, axis=1)
    res = np.concatenate(ws.resid) if ws.resid else np.array([np.nan])
    acc_hp, gyr_hp = _hp_std(streams.imu.acc), _hp_std(streams.imu.gyr)
    return {
        "gps_inlier_sigma_m": float(np.sqrt(np.median(gs.inlier_var_ne)))
        if gs.inlier_var_ne
        else np.nan,
        "gps_median_err_m": float(np.median(n)),
        "gps_frac_gt10m": float(np.mean(n > 10)),
        "gps_runs_gt10m_per_h": len(runs(n > 10)) / max(gs.hours, 1e-9),
        "gps_gross_per_h": len(gs.episodes) / max(gs.hours, 1e-9),
        "gps_mode2_frac": gs.mode2_fixes / max(gs.n_fixes, 1),
        "gps_held_frac": gs.held / max(gs.n_fixes, 1),
        "wheel_resid_robust_std": float(1.4826 * np.median(np.abs(res - np.median(res)))),
        "wheel_excursions_per_min": len(ws.events) / max(ws.moving_min, 1e-9),
        "acc_hp_std_x": float(acc_hp[0]),
        "acc_hp_std_y": float(acc_hp[1]),
        "acc_hp_std_z": float(acc_hp[2]),
        "gyr_hp_std_x": float(gyr_hp[0]),
        "gyr_hp_std_y": float(gyr_hp[1]),
        "gyr_hp_std_z": float(gyr_hp[2]),
    }
