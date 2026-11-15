"""Smooth truth trajectory derived from a real reference trajectory.

The dense reference is resampled to a uniform grid, low-pass filtered with a zero-phase
Butterworth filter, and fitted with quintic splines. The splines *define* the truth:
position, velocity, acceleration, attitude and body angular rate are exact at any time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.signal import butter, sosfiltfilt

from lase.data.raw import F64, I64
from lase.data.reference import Reference
from lase.geometry.euler import euler_zyx_to_matrix
from lase.utils.config import Config


class TrajectoryConfig(Config):
    """How the truth trajectory is derived from the reference."""

    resample_hz: float = 100.0
    lowpass_hz: float = 2.0  # removes interpolation artefacts; Segway motion is slower
    lowpass_order: int = 4
    spline_degree: int = 5


@dataclass(frozen=True)
class TruthState:
    """Truth at query times (body origin; NED world; body FRD)."""

    p_w: F64  # (Q, 3)
    v_w: F64  # (Q, 3)
    a_w: F64  # (Q, 3)
    rpy: F64  # (Q, 3) roll, pitch, yaw (rad, yaw unwrapped)
    C_wb: F64  # (Q, 3, 3)
    omega_b: F64  # (Q, 3) body angular rate (rad/s)
    omega_dot_b: F64  # (Q, 3) body angular acceleration (rad/s^2)


def euler_rates_to_body(rpy: F64, rpy_dot: F64) -> F64:
    """Body angular rate for ZYX Euler angles: omega_b = E(roll, pitch) @ [roll', pitch', yaw']."""
    r, p = rpy[..., 0], rpy[..., 1]
    dr, dp, dy = rpy_dot[..., 0], rpy_dot[..., 1], rpy_dot[..., 2]
    wx = dr - dy * np.sin(p)
    wy = dp * np.cos(r) + dy * np.sin(r) * np.cos(p)
    wz = -dp * np.sin(r) + dy * np.cos(r) * np.cos(p)
    out: F64 = np.stack([wx, wy, wz], axis=-1).astype(np.float64)
    return out


@dataclass(frozen=True)
class TruthTrajectory:
    """Spline truth trajectory over ``[t0_us, t1_us]``."""

    t0_us: int
    t1_us: int
    pos: BSpline  # seconds since t0 -> (3,)
    att: BSpline  # seconds since t0 -> unwrapped roll, pitch, yaw

    def _s(self, t_us: I64) -> F64:
        out: F64 = ((np.asarray(t_us, dtype=np.int64) - self.t0_us) * 1e-6).astype(np.float64)
        return out

    def contains(self, t_us: I64) -> np.ndarray:
        """Mask of times inside the trajectory span."""
        t = np.asarray(t_us)
        return (t >= self.t0_us) & (t <= self.t1_us)

    def omega_b(self, t_us: I64) -> F64:
        """Body angular rate (Q, 3)."""
        s = self._s(t_us)
        return euler_rates_to_body(self.att(s), self.att.derivative(1)(s))

    def at(self, t_us: I64) -> TruthState:
        """Full truth state at ``t_us`` (must lie inside the span)."""
        s = self._s(t_us)
        rpy = self.att(s)
        h = 1e-3  # central difference for angular acceleration (spline is smooth)
        w_plus = euler_rates_to_body(self.att(s + h), self.att.derivative(1)(s + h))
        w_minus = euler_rates_to_body(self.att(s - h), self.att.derivative(1)(s - h))
        return TruthState(
            p_w=self.pos(s),
            v_w=self.pos.derivative(1)(s),
            a_w=self.pos.derivative(2)(s),
            rpy=rpy,
            C_wb=euler_zyx_to_matrix(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
            omega_b=euler_rates_to_body(rpy, self.att.derivative(1)(s)),
            omega_dot_b=(w_plus - w_minus) / (2 * h),
        )

    def body_z_angle(self) -> BSpline:
        """Antiderivative of body z rate (rad) as a spline in seconds since t0 (FOG truth)."""
        grid = np.linspace(
            0.0, (self.t1_us - self.t0_us) * 1e-6, 1 + int((self.t1_us - self.t0_us) * 1e-4)
        )
        wz = self.omega_b(self.t0_us + np.round(grid * 1e6).astype(np.int64))[:, 2]
        spline: BSpline = make_interp_spline(grid, wz, k=3).antiderivative()
        return spline


def truth_from_reference(
    ref: Reference,
    config: TrajectoryConfig | None = None,
    t0_us: int | None = None,
    t1_us: int | None = None,
) -> TruthTrajectory:
    """Build a smooth truth trajectory from a session's valid dense reference.

    Args:
        ref: Session reference.
        config: Resampling and filtering settings.
        t0_us, t1_us: Optional window; defaults to the valid reference span.
    """
    cfg = config or TrajectoryConfig()
    m = ref.gt.valid
    t, p, rpy = ref.gt.t_us[m], ref.gt.p_w[m], ref.gt.rpy[m].copy()
    rpy = np.unwrap(rpy, axis=0)
    a = max(int(t[0]), t0_us or int(t[0]))
    b = min(int(t[-1]), t1_us or int(t[-1]))
    step = round(1e6 / cfg.resample_hz)
    grid = np.arange(a, b + 1, step, dtype=np.int64)
    s = (grid - a) * 1e-6
    tt = (t - a) * 1e-6
    P = np.stack([np.interp(s, tt, p[:, k]) for k in range(3)], axis=1)
    A = np.stack([np.interp(s, tt, rpy[:, k]) for k in range(3)], axis=1)
    sos = butter(cfg.lowpass_order, cfg.lowpass_hz, fs=cfg.resample_hz, output="sos")
    P = sosfiltfilt(sos, P, axis=0)
    A = sosfiltfilt(sos, A, axis=0)
    return TruthTrajectory(
        t0_us=int(grid[0]),
        t1_us=int(grid[-1]),
        pos=make_interp_spline(s, P, k=cfg.spline_degree),
        att=make_interp_spline(s, A, k=cfg.spline_degree),
    )
