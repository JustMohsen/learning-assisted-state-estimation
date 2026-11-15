"""Nominal navigation state and error-state layout (CONVENTIONS section 5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lase.filter.so3 import quat_exp, quat_mul, quat_normalize, quat_to_matrix

# Error-state index slices (16 core coordinates, fixed order).
P_ = slice(0, 3)
V_ = slice(3, 6)
TH = slice(6, 9)
BA = slice(9, 12)
BG = slice(12, 15)
BF = slice(15, 16)
WS = slice(16, 17)  # wheel speed scale (fixed when its prior and random walk are zero)
GE = slice(17, 19)  # GPS horizontal error, north/east (Gauss-Markov; inert when disabled)
GR = slice(19, 21)  # terrain height gradient dh/dN, dh/dE (random walk; inert when disabled)
N_ERR = 21


@dataclass(frozen=True)
class NavState:
    """Nominal state (batch-first, float64). Position/velocity are those of the IMU origin."""

    p_w: Tensor  # (B, 3)
    v_w: Tensor  # (B, 3)
    q_wb: Tensor  # (B, 4) body-to-world, Hamilton, scalar first
    b_acc: Tensor  # (B, 3) IMU axes
    b_gyr: Tensor  # (B, 3) IMU axes
    b_fog: Tensor  # (B, 1) FOG axis
    wheel_scale: Tensor  # (B, 1) released wheel speed / true speed
    gps_err: Tensor  # (B, 2) time-correlated GPS error, north/east (m)
    grade: Tensor  # (B, 2) terrain height gradient (dh/dN, dh/dE; rise/run, heading-independent)

    @property
    def C_wb(self) -> Tensor:
        """Body-to-world rotation (B, 3, 3)."""
        return quat_to_matrix(self.q_wb)

    def inject(self, dx: Tensor) -> NavState:
        """Apply an error-state correction (B, 16): additive except right-multiplicative attitude."""
        return NavState(
            p_w=self.p_w + dx[:, P_],
            v_w=self.v_w + dx[:, V_],
            q_wb=quat_normalize(quat_mul(self.q_wb, quat_exp(dx[:, TH]))),
            b_acc=self.b_acc + dx[:, BA],
            b_gyr=self.b_gyr + dx[:, BG],
            b_fog=self.b_fog + dx[:, BF],
            wheel_scale=self.wheel_scale + dx[:, WS],
            gps_err=self.gps_err + dx[:, GE],
            grade=self.grade + dx[:, GR],
        )


def diag_cov(sigmas: Tensor) -> Tensor:
    """Diagonal covariance (B, n, n) from standard deviations (B, n)."""
    return torch.diag_embed(sigmas**2)
