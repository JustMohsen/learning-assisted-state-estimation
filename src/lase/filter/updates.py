"""Measurement updates: generic gated Kalman update, wheel speed and GPS position.

Gain and covariance are analytical (linear solves, Joseph form). Rejected measurements
leave state and covariance unchanged. After injection, the attitude-error covariance is
reset consistently (first-order reset Jacobian).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import chi2
from torch import Tensor

from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterParams, GnssUpdateConfig, NonholonomicConfig, WheelUpdateConfig
from lase.filter.so3 import skew
from lase.filter.state import N_ERR, P_, TH, V_, WS, NavState


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Pre-update quantities (usable as causal context for learned heads)."""

    innovation: Tensor  # (B, d)
    S: Tensor  # (B, d, d)
    nis: Tensor  # (B,)
    accepted: Tensor  # (B,) bool


def gate_threshold(prob: float, dof: int) -> float:
    """Chi-square gate for a given acceptance probability."""
    return float(chi2.ppf(prob, dof))


def kalman_update(
    state: NavState,
    P: Tensor,
    r: Tensor,
    H: Tensor,
    R: Tensor,
    gate: float,
    apply: Tensor | None = None,
) -> tuple[NavState, Tensor, UpdateDiagnostics]:
    """Gated error-state Kalman update.

    Args:
        r: Innovation ``z - h(x)`` (B, d).
        H: Measurement Jacobian w.r.t. the error state (B, d, 16).
        R: Measurement covariance (B, d, d).
        gate: Chi-square threshold on the NIS.
        apply: Optional (B,) bool mask of batch elements that have this measurement.
    """
    PHt = P @ H.transpose(-1, -2)
    S = H @ PHt + R
    S = 0.5 * (S + S.transpose(-1, -2))
    Sinv_r = torch.linalg.solve(S, r[..., None])
    nis = (r[..., None] * Sinv_r).sum((-1, -2))
    ok = nis <= gate
    if apply is not None:
        ok = ok & apply
    K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)  # P H^T S^-1
    dx = (K @ r[..., None])[..., 0] * ok[:, None]
    eye = torch.eye(N_ERR, dtype=P.dtype)
    IKH = eye - K @ H
    P_post = IKH @ P @ IKH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    # reset Jacobian for the attitude error after injection
    G = eye.repeat(P.shape[0], 1, 1)
    G[:, TH, TH] = torch.eye(3, dtype=P.dtype) - 0.5 * skew(dx[:, TH])
    P_post = G @ P_post @ G.transpose(-1, -2)
    P_post = 0.5 * (P_post + P_post.transpose(-1, -2))
    P_new = torch.where(ok[:, None, None], P_post, P)
    return state.inject(dx), P_new, UpdateDiagnostics(r, S, nis, ok)


def wheel_prediction(state: NavState, omega_b: Tensor, params: FilterParams) -> Tensor:
    """Forward speed of the axle centre (B,) before the wheel scale is applied."""
    C = state.C_wb
    v_b = (C.transpose(-1, -2) @ state.v_w[..., None])[..., 0]
    v_axle: Tensor = v_b - torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    return v_axle[:, 0]


def update_wheel(
    state: NavState,
    P: Tensor,
    left: Tensor,
    right: Tensor,
    omega_b: Tensor,
    params: FilterParams,
    cfg: WheelUpdateConfig,
    learned: HeadOutputs,
    apply: Tensor | None = None,
) -> tuple[NavState, Tensor, UpdateDiagnostics]:
    """Scalar forward-speed update (mean of left/right wheel speeds).

    ``omega_b`` (latest corrected body rate) is treated as an exogenous input; its shared
    gyro noise with propagation is a disclosed approximation.
    """
    s = state.wheel_scale[:, 0]
    z = 0.5 * (left + right)
    v_fwd = wheel_prediction(state, omega_b, params)
    h = s * v_fwd + learned.wheel_residual[:, 0]
    C = state.C_wb
    v_b = (C.transpose(-1, -2) @ state.v_w[..., None])[..., 0]
    B = P.shape[0]
    H = torch.zeros(B, 1, N_ERR, dtype=P.dtype)
    H[:, 0, V_] = s[:, None] * C[:, :, 0]  # e_x^T C^T
    H[:, 0, TH] = s[:, None] * skew(v_b)[:, 0, :]  # e_x^T [C^T v]x
    H[:, 0, WS] = v_fwd
    R = (cfg.sigma_mps**2 * learned.wheel_var_mult[:, 0])[:, None, None]
    return kalman_update(state, P, (z - h)[:, None], H, R, gate_threshold(cfg.gate_prob, 1), apply)


def axle_velocity_body(state: NavState, omega_b: Tensor, params: FilterParams) -> Tensor:
    """Body-frame velocity of the axle centre (B, 3)."""
    C = state.C_wb
    v_b = (C.transpose(-1, -2) @ state.v_w[..., None])[..., 0]
    out: Tensor = v_b - torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    return out


def update_nonholonomic(
    state: NavState,
    P: Tensor,
    omega_b: Tensor,
    params: FilterParams,
    cfg: NonholonomicConfig,
    learned: HeadOutputs | None = None,
) -> tuple[NavState, Tensor, UpdateDiagnostics]:
    """Soft pseudo-measurement: the axle does not slide sideways (lateral body velocity ~ 0).

    Lateral is along the axle (body y), so it is unaffected by pitch. The optional vertical
    term (body z velocity ~ 0) is wrong when the Segway pitches (body z velocity = v sin(pitch))
    and is off by default. ``learned.nhc_var_mult`` scales the lateral variance (B/C methods).
    """
    C = state.C_wb
    v_b = (C.transpose(-1, -2) @ state.v_w[..., None])[..., 0]
    rows = [1, 2] if cfg.vertical_enabled else [1]
    h = axle_velocity_body(state, omega_b, params)[:, rows]
    B, d = P.shape[0], len(rows)
    H = torch.zeros(B, d, N_ERR, dtype=P.dtype)
    H[:, :, V_] = C.transpose(-1, -2)[:, rows, :]
    H[:, :, TH] = skew(v_b)[:, rows, :]
    var = [cfg.sigma_lateral_mps**2, cfg.sigma_vertical_mps**2][:d]
    R = torch.diag(torch.tensor(var, dtype=P.dtype)).expand(B, d, d).clone()
    if learned is not None:
        R[:, 0, 0] = R[:, 0, 0] * learned.nhc_var_mult[:, 0]
    return kalman_update(state, P, -h, H, R, gate_threshold(cfg.gate_prob, d))


def gnss_prediction(state: NavState, params: FilterParams) -> Tensor:
    """Predicted horizontal antenna position (B, 2)."""
    d = params.l_gps - params.l_imu
    return (state.p_w + (state.C_wb @ d))[:, :2]


def update_gnss(
    state: NavState,
    P: Tensor,
    z_ne: Tensor,
    params: FilterParams,
    cfg: GnssUpdateConfig,
    learned: HeadOutputs,
    apply: Tensor | None = None,
) -> tuple[NavState, Tensor, UpdateDiagnostics]:
    """Horizontal antenna-position update (2-dof)."""
    d = params.l_gps - params.l_imu
    C = state.C_wb
    B = P.shape[0]
    H = torch.zeros(B, 2, N_ERR, dtype=P.dtype)
    H[:, :, P_] = torch.eye(3, dtype=P.dtype)[:2]
    H[:, :, TH] = -(C @ skew(d))[:, :2, :]
    R = torch.diag_embed(cfg.sigma_m**2 * learned.gnss_var_mult)
    r = z_ne - gnss_prediction(state, params)
    return kalman_update(state, P, r, H, R, gate_threshold(cfg.gate_prob, 2), apply)
