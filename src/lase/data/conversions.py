"""Sensor-specific conversions (FOG angle handling, wheel kinematics).

Functions marked *offline* may interpolate between samples on both sides of a query
time; they are for labels, validation and analysis, never for estimator inputs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def unwrap_fog(angle: F64) -> F64:
    """Unwrap the FOG's integrated heading angle (released wrapped to [0, 2*pi))."""
    out: F64 = np.unwrap(np.asarray(angle, dtype=np.float64))
    return out


def fog_increments(angle_unwrapped: F64) -> F64:
    """Angle change between consecutive FOG samples (causal; first element is 0)."""
    inc = np.zeros_like(angle_unwrapped)
    inc[1:] = np.diff(angle_unwrapped)
    return inc


def interp_at(t_us: I64, values: F64, t_query: I64) -> F64:
    """*Offline* linear interpolation of ``values`` (N, ...) sampled at ``t_us`` onto ``t_query``.

    Queries outside the sample span return NaN.
    """
    t = (t_us - t_us[0]).astype(np.float64)
    q = (np.asarray(t_query) - t_us[0]).astype(np.float64)
    v = np.asarray(values, dtype=np.float64)
    flat = v.reshape(len(t), -1)
    out = np.stack(
        [np.interp(q, t, flat[:, k], left=np.nan, right=np.nan) for k in range(flat.shape[1])],
        axis=-1,
    )
    res: F64 = out.reshape(len(q), *v.shape[1:])
    return res


def rate_on_grid(t_us: I64, angle: F64, grid_us: I64) -> F64:
    """*Offline* angular rate (rad/s) on a regular grid, by central differences of interpolated angle."""
    a = interp_at(t_us, angle, grid_us)
    tg = grid_us.astype(np.float64) * 1e-6
    rate: F64 = np.gradient(a, tg)
    return rate


def wheel_forward_speed(left: F64, right: F64) -> F64:
    """Speed at the wheelbase centre (paper Eq. 4)."""
    out: F64 = 0.5 * (np.asarray(left) + np.asarray(right))
    return out


def wheel_yaw_rate(left: F64, right: F64, track_width_m: float) -> F64:
    """Yaw rate about body z (down) from differential wheel speeds: (left - right) / track.

    With z pointing down, positive yaw is a right turn, which needs the left wheel faster.
    """
    out: F64 = (np.asarray(left) - np.asarray(right)) / track_width_m
    return out
