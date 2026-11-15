"""Validation-only strapdown integrator (NumPy).

An independent, deliberately simple implementation used to validate the simulator
(and later to cross-check the PyTorch filter's propagation). It is not part of the
estimator: the filter has its own implementation (Phase 3).
"""

from __future__ import annotations

import numpy as np

from lase.data.raw import F64, I64
from lase.geometry.constants import GRAVITY_MPS2


def so3_exp(phi: F64) -> F64:
    """Rotation matrix ``Exp(phi)`` for a rotation vector ``phi`` (3,)."""
    angle = float(np.linalg.norm(phi))
    K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    if angle < 1e-12:
        small: F64 = np.eye(3) + K
        return small
    k = K / angle
    out: F64 = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return out


def strapdown(t_us: I64, acc: F64, gyr: F64, p0: F64, v0: F64, C0: F64) -> tuple[F64, F64, F64]:
    """Integrate IMU samples (sensor axes = body axes) in NED with constant gravity.

    Uses midpoint (trapezoidal) angular rate and specific force per interval, with the
    attitude at the interval midpoint for rotating specific force.

    Returns:
        Positions (N, 3), velocities (N, 3) and attitudes (N, 3, 3) at ``t_us``.
    """
    n = len(t_us)
    g = np.array([0.0, 0.0, GRAVITY_MPS2])
    p, v, C = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3))
    p[0], v[0], C[0] = p0, v0, C0
    for k in range(n - 1):
        dt = (t_us[k + 1] - t_us[k]) * 1e-6
        w = 0.5 * (gyr[k] + gyr[k + 1])
        f = 0.5 * (acc[k] + acc[k + 1])
        C_mid = C[k] @ so3_exp(0.5 * w * dt)
        a = C_mid @ f + g
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a * dt * dt
        v[k + 1] = v[k] + a * dt
        C[k + 1] = C[k] @ so3_exp(w * dt)
    return p, v, C
