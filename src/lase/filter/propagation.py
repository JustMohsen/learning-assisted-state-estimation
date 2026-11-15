"""Inertial propagation of the nominal state and error covariance.

One step covers a completed IMU interval ``[t_k, t_{k+1}]``. Angular motion about body x/y
comes from the IMU gyro; about body z it is the inverse-variance combination of the FOG
increment and the IMU gyro z (FOG dominates; IMU-only if no FOG data in the interval).
Midpoint integration matches the validated strapdown reference (``lase.sim.strapdown``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterParams, ProcessNoise
from lase.filter.so3 import (
    quat_exp,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
    skew,
    so3_exp,
    so3_right_jacobian,
)
from lase.filter.state import BA, BF, BG, N_ERR, P_, TH, V_, WS, NavState


@dataclass(frozen=True)
class ImuInterval:
    """Inputs for one propagation step (batch-first)."""

    dt: Tensor  # (B,) seconds
    acc0: Tensor  # (B, 3) IMU axes, sample at t_k
    acc1: Tensor  # (B, 3) sample at t_{k+1}
    gyr0: Tensor  # (B, 3)
    gyr1: Tensor  # (B, 3)
    fog_dangle: Tensor  # (B,) released FOG angle change over the interval (rad, raw sign)
    fog_valid: Tensor  # (B,) 1.0 if FOG data covers the interval, else 0.0


@dataclass(frozen=True)
class PropagationDiagnostics:
    """Quantities useful for outputs and learned-model context."""

    omega_b: Tensor  # (B, 3) corrected mean body rate over the interval (z fused)
    f_b: Tensor  # (B, 3) corrected mean specific force (body axes)
    fog_weight: Tensor  # (B,)
    F: Tensor  # (B, 16, 16) error-state transition used for the covariance


def propagate(
    state: NavState,
    P: Tensor,
    iv: ImuInterval,
    params: FilterParams,
    noise: ProcessNoise,
    learned: HeadOutputs,
) -> tuple[NavState, Tensor, PropagationDiagnostics]:
    """Propagate mean and covariance over one IMU interval."""
    dt = iv.dt
    dt_ = dt[:, None]
    Cb = params.C_b_imu
    f0 = (iv.acc0 - state.b_acc) @ Cb.T
    f1 = (iv.acc1 - state.b_acc) @ Cb.T
    w0 = (iv.gyr0 - state.b_gyr) @ Cb.T
    w1 = (iv.gyr1 - state.b_gyr) @ Cb.T
    f = 0.5 * (f0 + f1)
    C = state.C_wb
    w_ie_b = C.transpose(-1, -2) @ params.earth_rate_w  # Earth rate in body axes (B, 3)
    w = 0.5 * (w0 + w1) - w_ie_b

    qg, qf = noise.gyr_noise_density**2, noise.fog_angle_random_walk**2
    wf = iv.fog_valid * (qg / (qg + qf))  # FOG weight on body z
    dth_fog = params.fog_sign * iv.fog_dangle - state.b_fog[:, 0] * dt - w_ie_b[:, 2] * dt
    dth_z = wf * dth_fog + (1 - wf) * w[:, 2] * dt
    phi = torch.stack([w[:, 0] * dt, w[:, 1] * dt, dth_z], dim=-1)

    E_half = so3_exp(0.5 * phi)
    C_mid = C @ E_half
    a = (C_mid @ f[..., None])[..., 0] + params.gravity_w
    p_new = state.p_w + state.v_w * dt_ + 0.5 * a * dt_**2
    v_new = state.v_w + a * dt_
    q_new = quat_normalize(quat_mul(state.q_wb, quat_exp(phi)))
    drift = learned.bias_drift
    new = NavState(
        p_w=p_new,
        v_w=v_new,
        q_wb=q_new,
        b_acc=state.b_acc + drift[:, 0:3] * dt_,
        b_gyr=state.b_gyr + drift[:, 3:6] * dt_,
        b_fog=state.b_fog,
        wheel_scale=state.wheel_scale,
    )

    # --- error-state transition F (first order) ---
    B = P.shape[0]
    eye3 = torch.eye(3, dtype=P.dtype).expand(B, 3, 3)
    dt3 = dt[:, None, None]
    # rotation increment sensitivity: d phi = G dx (gyro bias on x/y/(1-w)z, FOG bias on w z)
    G = torch.zeros(B, 3, N_ERR, dtype=P.dtype)
    gyro_rows = torch.stack([torch.ones_like(wf), torch.ones_like(wf), 1 - wf], dim=-1)
    G[:, :, BG] = -(gyro_rows[..., None] * Cb) * dt3
    G[:, 2, 15] = -wf * dt
    G[:, :, TH] = -skew(w_ie_b) * dt3  # Earth-rate projection depends on attitude
    Jr = so3_right_jacobian(phi)
    Jr_half = so3_right_jacobian(0.5 * phi)
    A_mid = -C_mid @ skew(f)  # d a / d (rotation error at the midpoint)
    F = torch.eye(N_ERR, dtype=P.dtype).repeat(B, 1, 1)
    # acceleration sensitivity: attitude at start, accel bias, and gyro/FOG bias via the midpoint
    dA = torch.zeros(B, 3, N_ERR, dtype=P.dtype)
    dA[:, :, TH] = A_mid @ E_half.transpose(-1, -2)
    dA[:, :, BA] = -C_mid @ Cb
    dA = dA + A_mid @ (0.5 * Jr_half @ G)
    F[:, P_, :] = F[:, P_, :] + 0.5 * dA * dt3**2
    F[:, P_, V_] = F[:, P_, V_] + eye3 * dt3
    F[:, V_, :] = F[:, V_, :] + dA * dt3
    F[:, TH, :] = Jr @ G
    F[:, TH, TH] = so3_exp(phi).transpose(-1, -2) + (Jr @ G)[:, :, TH]

    # --- discrete process noise Q ---
    qa = noise.acc_noise_density**2
    Q = torch.zeros(B, N_ERR, N_ERR, dtype=P.dtype)
    Q[:, P_, P_] = eye3 * (qa * dt3**3 / 3)
    Q[:, P_, V_] = eye3 * (qa * dt3**2 / 2)
    Q[:, V_, P_] = eye3 * (qa * dt3**2 / 2)
    Q[:, V_, V_] = eye3 * (qa * dt3)
    th_var = torch.stack([qg * dt, qg * dt, ((1 - wf) ** 2) * qg * dt + (wf**2) * qf * dt], dim=-1)
    Q[:, TH, TH] = torch.diag_embed(th_var)
    Q[:, BA, BA] = eye3 * (noise.acc_bias_rw**2 * dt3)
    Q[:, BG, BG] = eye3 * (noise.gyr_bias_rw**2 * dt3)
    Q[:, BF, BF] = (noise.fog_bias_rw**2 * dt)[:, None, None]
    Q[:, WS, WS] = (noise.wheel_scale_rw**2 * dt)[:, None, None]

    P_new = F @ P @ F.transpose(-1, -2) + Q
    P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
    omega = torch.stack([w[:, 0], w[:, 1], dth_z / dt], dim=-1)
    return new, P_new, PropagationDiagnostics(omega_b=omega, f_b=f, fog_weight=wf, F=F)


def boxminus(a: NavState, b: NavState) -> Tensor:
    """Error-state difference ``a ⊟ b`` (B, 17): additive except attitude Log(C_b^T C_a)."""
    from lase.filter.so3 import so3_log

    dth = so3_log(b.C_wb.transpose(-1, -2) @ a.C_wb)
    return torch.cat(
        [
            a.p_w - b.p_w,
            a.v_w - b.v_w,
            dth,
            a.b_acc - b.b_acc,
            a.b_gyr - b.b_gyr,
            a.b_fog - b.b_fog,
            a.wheel_scale - b.wheel_scale,
        ],
        dim=-1,
    )


__all__ = ["ImuInterval", "PropagationDiagnostics", "boxminus", "propagate", "quat_to_matrix"]
