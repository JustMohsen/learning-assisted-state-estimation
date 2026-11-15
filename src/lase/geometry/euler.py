"""Euler-angle conventions used by the NCLT dataset (NumPy, data side).

NCLT (paper Sec. 3) composes ``R_ij = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, mapping vectors
from frame ``j`` into frame ``i``. With our naming this is ``C_ij``; for a ground-truth
pose it is ``C_wb`` (body to local NED).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def rot_x(a: F64 | float) -> F64:
    """Active rotation(s) about x by angle(s) ``a`` (rad). Shape ``(..., 3, 3)``."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(*a.shape, 3, 3)


def rot_y(a: F64 | float) -> F64:
    """Active rotation(s) about y by angle(s) ``a`` (rad). Shape ``(..., 3, 3)``."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(*a.shape, 3, 3)


def rot_z(a: F64 | float) -> F64:
    """Active rotation(s) about z by angle(s) ``a`` (rad). Shape ``(..., 3, 3)``."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, -s, z, s, c, z, z, z, o], axis=-1).reshape(*a.shape, 3, 3)


def euler_zyx_to_matrix(roll: F64 | float, pitch: F64 | float, yaw: F64 | float) -> F64:
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (NCLT convention). Shape ``(..., 3, 3)``."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_euler_zyx(C: F64) -> tuple[F64, F64, F64]:
    """Inverse of :func:`euler_zyx_to_matrix` (roll, pitch, yaw in rad; pitch in [-pi/2, pi/2])."""
    roll = np.arctan2(C[..., 2, 1], C[..., 2, 2])
    pitch = -np.arcsin(np.clip(C[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(C[..., 1, 0], C[..., 0, 0])
    return roll, pitch, yaw


def wrap_angle(a: F64 | float) -> F64:
    """Wrap angle(s) to ``(-pi, pi]``."""
    w = np.mod(np.asarray(a, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    return np.where(w == -np.pi, np.pi, w)
