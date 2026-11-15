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
from lase.filter.state import GE, GR, N_ERR, P_, TH, V_, WS, NavState


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


def axle_velocity_world(state: NavState, omega_b: Tensor, params: FilterParams) -> Tensor:
    """World-frame velocity of the axle centre (B, 3): v_IMU - C (omega x l_imu)."""
    lever = torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    out: Tensor = state.v_w - (state.C_wb @ lever[..., None])[..., 0]
    return out


def rolling_direction(C: Tensor) -> tuple[Tensor, Tensor]:
    """Horizontal unit direction (B, 3) perpendicular to the axle, and the norm used.

    ``f = normalize(axle_w x down)`` with ``axle_w = C e_y``. For a level body this is body x.
    It depends on attitude (well observed via gravity), not on the vertical velocity estimate,
    and is independent of body pitch. Assumes locally level ground (campus slopes of a few
    percent change the rolling speed by < 0.1%).
    """
    axle = C[:, :, 1]
    down = torch.tensor([0.0, 0.0, 1.0], dtype=C.dtype).expand_as(axle)
    n = torch.linalg.cross(axle, down)
    norm = n.norm(dim=-1, keepdim=True)
    return n / norm, norm


def rolling_speed_with_jacobian(
    state: NavState, omega_b: Tensor, params: FilterParams
) -> tuple[Tensor, Tensor, Tensor]:
    """Rolling speed V = f . v_axle (B,), the direction f (B, 3), and dV/dtheta (B, 3).

    d f / d theta: n = (C e_y) x d, dn = [d]x C [e_y]x dtheta, df = (I - f f^T)/|n| dn;
    v_axle = v - C (omega x l): d v_axle / d theta = C [omega x l]x.
    """
    C = state.C_wb
    f, norm = rolling_direction(C)
    u = axle_velocity_world(state, omega_b, params)
    v_roll = (f * u).sum(-1)
    B = C.shape[0]
    eye3 = torch.eye(3, dtype=C.dtype).expand(B, 3, 3)
    e_y = torch.tensor([0.0, 1.0, 0.0], dtype=C.dtype).expand(B, 3)
    down = torch.tensor([0.0, 0.0, 1.0], dtype=C.dtype).expand(B, 3)
    lever = torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    proj = (eye3 - f[..., None] * f[..., None, :]) / norm[..., None]
    df_dth = proj @ skew(down) @ C @ skew(e_y)
    du_dth = C @ skew(lever)
    dv = (u[:, None, :] @ df_dth)[:, 0] + (f[:, None, :] @ du_dth)[:, 0]
    return v_roll, f, dv


def terrain_following_row(
    state: NavState, omega_b: Tensor, params: FilterParams
) -> tuple[Tensor, Tensor]:
    """Terrain-following residual h = u_D + g_N u_N + g_E u_E (B,) and its Jacobian (B, N_ERR).

    The axle stays on the ground, h(N, E): its down-velocity is -(grad h . horizontal velocity).
    The gradient g is estimated, so slopes are allowed; only motion off the terrain is penalised.
    u = v - C (omega x l), so du/dv = I and du/dtheta = C [omega x l]x.
    """
    C = state.C_wb
    u = axle_velocity_world(state, omega_b, params)
    g = state.grade
    a = torch.cat([g, torch.ones_like(g[:, :1])], dim=-1)  # d h / d u
    lever = torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    H = torch.zeros(C.shape[0], N_ERR, dtype=C.dtype)
    H[:, V_] = a
    H[:, TH] = (a[:, None, :] @ C @ skew(lever))[:, 0]
    H[:, GR] = u[:, :2]
    return (a * u).sum(-1), H


def wheel_prediction(state: NavState, omega_b: Tensor, params: FilterParams) -> Tensor:
    """Rolling speed of the axle centre (B,), before wheel scale and encoder pitch-rate term."""
    f, _ = rolling_direction(state.C_wb)
    out: Tensor = (f * axle_velocity_world(state, omega_b, params)).sum(-1)
    return out


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
    v_roll, f, dvroll_dth = rolling_speed_with_jacobian(state, omega_b, params)
    # Encoders measure wheel rotation relative to the chassis: w_enc = V/R + w_pitch, so the
    # reported speed contains R * q (q = body pitch rate, gyro y; Phase 3 review).
    h = s * v_roll + cfg.encoder_pitch_coef_m * omega_b[:, 1] + learned.wheel_residual[:, 0]
    B = P.shape[0]
    H = torch.zeros(B, 1, N_ERR, dtype=P.dtype)
    H[:, 0, V_] = s[:, None] * f
    H[:, 0, TH] = s[:, None] * dvroll_dth
    H[:, 0, WS] = v_roll
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
    """Soft pseudo-measurements on the axle velocity.

    Lateral (always): velocity along the axle (body y) ~ 0; unaffected by pitch.
    Vertical (``cfg.vertical_mode``): "body" constrains body-z velocity (biased by
    v sin(pitch) when the Segway pitches); "level" constrains world-vertical velocity
    (pitch-independent; road slopes fall within its sigma); "off" omits it.
    ``learned.nhc_var_mult`` scales the lateral variance (B/C methods).
    """
    C = state.C_wb
    Ct = C.transpose(-1, -2)
    v_b = (Ct @ state.v_w[..., None])[..., 0]
    B = P.shape[0]
    lever = torch.linalg.cross(omega_b, params.l_imu.expand_as(omega_b))
    h_rows = [axle_velocity_body(state, omega_b, params)[:, 1]]
    H_rows = []
    Hl = torch.zeros(B, N_ERR, dtype=P.dtype)
    Hl[:, V_] = Ct[:, 1, :]
    Hl[:, TH] = skew(v_b)[:, 1, :]
    H_rows.append(Hl)
    if cfg.vertical_mode == "body":
        h_rows.append(axle_velocity_body(state, omega_b, params)[:, 2])
        Hv = torch.zeros(B, N_ERR, dtype=P.dtype)
        Hv[:, V_] = Ct[:, 2, :]
        Hv[:, TH] = skew(v_b)[:, 2, :]
        H_rows.append(Hv)
    elif cfg.vertical_mode == "level":
        # world-down axle velocity: v_D - (C (omega x l))_D; d/dtheta = (C [omega x l]x)_D
        h_rows.append(axle_velocity_world(state, omega_b, params)[:, 2])
        Hv = torch.zeros(B, N_ERR, dtype=P.dtype)
        Hv[:, 5] = 1.0
        Hv[:, TH] = (C @ skew(lever))[:, 2, :]
        H_rows.append(Hv)
    elif cfg.vertical_mode == "grade":
        hv, Hv = terrain_following_row(state, omega_b, params)
        h_rows.append(hv)
        H_rows.append(Hv)
    h = torch.stack(h_rows, dim=-1)
    H = torch.stack(H_rows, dim=1)
    d = h.shape[1]
    var = [cfg.sigma_lateral_mps**2, cfg.sigma_vertical_mps**2][:d]
    R = torch.diag(torch.tensor(var, dtype=P.dtype)).expand(B, d, d).clone()
    if learned is not None:
        R[:, 0, 0] = R[:, 0, 0] * learned.nhc_var_mult[:, 0]
    return kalman_update(state, P, -h, H, R, gate_threshold(cfg.gate_prob, d))


def gnss_prediction(state: NavState, params: FilterParams) -> Tensor:
    """Predicted GPS fix (B, 2): horizontal antenna position plus the correlated GPS error."""
    d = params.l_gps - params.l_imu
    out: Tensor = (state.p_w + (state.C_wb @ d))[:, :2] + state.gps_err
    return out


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
    H[:, :, GE] = torch.eye(2, dtype=P.dtype)
    sigma = cfg.error_white_sigma_m if cfg.error_state else cfg.sigma_m
    R = torch.diag_embed(sigma**2 * learned.gnss_var_mult)
    r = z_ne - gnss_prediction(state, params)
    return kalman_update(state, P, r, H, R, gate_threshold(cfg.gate_prob, 2), apply)
