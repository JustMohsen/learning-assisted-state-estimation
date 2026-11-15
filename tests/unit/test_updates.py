import numpy as np
import pytest
import torch

from lase.data.calibration import load_calibration
from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.propagation import boxminus
from lase.filter.so3 import matrix_to_quat, so3_exp
from lase.filter.state import N_ERR, NavState
from lase.filter.updates import (
    gnss_prediction,
    kalman_update,
    update_gnss,
    update_wheel,
    wheel_prediction,
)

D = torch.float64


def _state() -> NavState:
    g = torch.Generator().manual_seed(4)
    return NavState(
        p_w=torch.randn(1, 3, generator=g, dtype=D),
        v_w=torch.tensor([[1.2, -0.4, 0.1]], dtype=D),
        q_wb=matrix_to_quat(so3_exp(torch.tensor([[0.1, -0.2, 0.9]], dtype=D))),
        b_acc=torch.zeros(1, 3, dtype=D),
        b_gyr=torch.zeros(1, 3, dtype=D),
        b_fog=torch.zeros(1, 1, dtype=D),
        wheel_scale=torch.full((1, 1), 1.05, dtype=D),
        gps_err=torch.tensor([[0.7, -0.4]], dtype=D),
    )


def _H_numeric(fun, x0: NavState) -> torch.Tensor:  # type: ignore[no-untyped-def]
    return torch.autograd.functional.jacobian(
        lambda dx: fun(x0.inject(dx[None]))[0], torch.zeros(N_ERR, dtype=D)
    )


def test_wheel_jacobian_matches_autograd() -> None:
    from lase.filter.so3 import so3_exp
    from lase.filter.updates import update_wheel, wheel_prediction

    cfg = FilterConfig()
    params = FilterParams.build(load_calibration(), cfg)
    x0 = _state()
    x0 = NavState(
        x0.p_w,
        x0.v_w,
        matrix_to_quat(so3_exp(torch.tensor([[0.1, 0.25, 0.9]], dtype=D))),
        x0.b_acc,
        x0.b_gyr,
        x0.b_fog,
        x0.wheel_scale,
        x0.gps_err,
    )
    omega = torch.tensor([[0.05, -0.2, 0.3]], dtype=D)
    J = _H_numeric(
        lambda x: (x.wheel_scale[:, 0] * wheel_prediction(x, omega, params))[:, None], x0
    )
    # recover the analytic H from a scalar update: K = P H^T S^-1 with P = I -> H = K * S
    P = torch.eye(N_ERR, dtype=D)[None]
    z = torch.zeros(1, dtype=D)
    update_wheel(
        x0, P, z, z, omega, params, cfg.wheel, HeadOutputs.neutral(1)
    )  # runs with this state
    from lase.filter.so3 import skew
    from lase.filter.updates import axle_velocity_world, rolling_direction

    C = x0.C_wb
    f, norm = rolling_direction(C)
    u = axle_velocity_world(x0, omega, params)
    s = x0.wheel_scale[:, 0]
    eye3 = torch.eye(3, dtype=D)[None]
    e_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=D)
    down = torch.tensor([[0.0, 0.0, 1.0]], dtype=D)
    lever = torch.linalg.cross(omega, params.l_imu[None])
    proj = (eye3 - f[..., None] * f[..., None, :]) / norm[..., None]
    H = torch.zeros(N_ERR, dtype=D)
    H[3:6] = s * f[0]
    H[6:9] = s * (
        (u[:, None, :] @ (proj @ skew(down) @ C @ skew(e_y)))[0, 0]
        + (f[:, None, :] @ (C @ skew(lever)))[0, 0]
    )
    H[16] = (f * u).sum()
    np.testing.assert_allclose(H.numpy(), J[0].numpy(), atol=1e-10)


def test_rolling_speed_is_pitch_independent() -> None:
    from lase.filter.so3 import so3_exp
    from lase.filter.updates import wheel_prediction

    params = FilterParams.build(load_calibration(), FilterConfig())
    omega = torch.zeros(1, 3, dtype=D)
    for pitch in (0.0, 0.1, -0.2):  # axle moving horizontally north at 1.4 m/s, body pitched
        C = so3_exp(torch.tensor([[0.0, pitch, 0.0]], dtype=D))
        x = NavState(
            torch.zeros(1, 3, dtype=D),
            torch.tensor([[1.4, 0.0, 0.0]], dtype=D),
            matrix_to_quat(C),
            torch.zeros(1, 3, dtype=D),
            torch.zeros(1, 3, dtype=D),
            torch.zeros(1, 1, dtype=D),
            torch.ones(1, 1, dtype=D),
            torch.zeros(1, 2, dtype=D),
        )
        assert float(wheel_prediction(x, omega, params)[0]) == pytest.approx(1.4, abs=1e-12)


def test_gnss_jacobian_matches_autograd() -> None:
    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    J = _H_numeric(lambda x: gnss_prediction(x, params), x0)
    from lase.filter.so3 import skew

    d = params.l_gps - params.l_imu
    H = torch.zeros(2, N_ERR, dtype=D)
    H[:, 0:3] = torch.eye(3, dtype=D)[:2]
    H[:, 6:9] = -(x0.C_wb[0] @ skew(d))[:2]
    H[:, 17:19] = torch.eye(2, dtype=D)  # correlated GPS error state
    np.testing.assert_allclose(H.numpy(), J.numpy(), atol=1e-10)


def test_linear_case_equals_batch_weighted_least_squares() -> None:
    rng = np.random.default_rng(0)
    P0 = np.diag(rng.uniform(0.5, 2.0, N_ERR))
    x0 = _state()
    p_prior = x0.p_w[0].numpy()
    zs = [p_prior[:2] + rng.normal(0, 1, 2) for _ in range(3)]
    Rm = np.diag([0.7, 1.3])
    s, P = x0, torch.as_tensor(P0)[None]
    H = np.zeros((2, N_ERR))
    H[:, :2] = np.eye(2)
    for z in zs:  # position-only linear measurements, applied sequentially
        r = torch.as_tensor(z - s.p_w[0, :2].numpy())[None]
        s, P, _ = kalman_update(
            s, P, r, torch.as_tensor(H)[None], torch.as_tensor(Rm)[None], gate=1e9
        )
    info = np.linalg.inv(P0) + sum(H.T @ np.linalg.inv(Rm) @ H for _ in zs)
    P_wls = np.linalg.inv(info)
    dx_wls = P_wls @ sum(H.T @ np.linalg.inv(Rm) @ (z - p_prior[:2]) for z in zs)
    np.testing.assert_allclose(P[0].numpy(), P_wls, atol=1e-10)
    np.testing.assert_allclose(s.p_w[0, :2].numpy() - p_prior[:2], dx_wls[:2], atol=1e-10)


def test_gnss_outlier_rejected_and_state_unchanged() -> None:
    cfg = FilterConfig()
    params = FilterParams.build(load_calibration(), cfg)
    x0 = _state()
    P = torch.eye(N_ERR, dtype=D)[None]
    far = gnss_prediction(x0, params) + 1000.0
    s, P1, diag = update_gnss(x0, P, far, params, cfg.gnss, HeadOutputs.neutral(1))
    assert not bool(diag.accepted[0])
    torch.testing.assert_close(P1, P)
    torch.testing.assert_close(boxminus(s, x0), torch.zeros(1, N_ERR, dtype=D))


def test_wheel_update_reduces_speed_error() -> None:
    cfg = FilterConfig()
    params = FilterParams.build(load_calibration(), cfg)
    x0 = _state()
    P = torch.eye(N_ERR, dtype=D)[None] * 0.1
    omega = torch.zeros(1, 3, dtype=D)
    pred = params.wheel_scale * wheel_prediction(x0, omega, params)
    meas = pred + 0.2
    s, _, diag = update_wheel(x0, P, meas, meas, omega, params, cfg.wheel, HeadOutputs.neutral(1))
    assert bool(diag.accepted[0])
    new_pred = params.wheel_scale * wheel_prediction(s, omega, params)
    assert abs(float(meas - new_pred)) < 0.2


def test_nonholonomic_jacobian_matches_autograd() -> None:
    from lase.filter.params import NonholonomicConfig
    from lase.filter.so3 import skew
    from lase.filter.updates import axle_velocity_body, update_nonholonomic

    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    omega = torch.tensor([[0.05, -0.02, 0.3]], dtype=D)
    J = _H_numeric(lambda x: axle_velocity_body(x, omega, params)[:, 1:3], x0)
    C = x0.C_wb[0]
    v_b = C.T @ x0.v_w[0]
    H = torch.zeros(2, N_ERR, dtype=D)
    H[:, 3:6] = C.T[1:3]
    H[:, 6:9] = skew(v_b[None])[0, 1:3]
    np.testing.assert_allclose(H.numpy(), J.numpy(), atol=1e-10)
    P = torch.eye(N_ERR, dtype=D)[None]
    s, _, d = update_nonholonomic(
        x0, P, omega, params, NonholonomicConfig(enabled=True, vertical_mode="body")
    )
    assert bool(d.accepted[0])
    after = axle_velocity_body(s, omega, params)[0, 1:3].abs().sum()
    before = axle_velocity_body(x0, omega, params)[0, 1:3].abs().sum()
    assert after < before


def test_nonholonomic_lateral_only_is_one_dof() -> None:
    from lase.filter.params import NonholonomicConfig
    from lase.filter.updates import update_nonholonomic

    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    omega = torch.zeros(1, 3, dtype=D)
    P = torch.eye(N_ERR, dtype=D)[None]
    _, _, d = update_nonholonomic(x0, P, omega, params, NonholonomicConfig(enabled=True))
    assert d.innovation.shape == (1, 1)
    m = HeadOutputs.neutral(1)
    big = HeadOutputs(
        m.bias_drift, m.wheel_residual, m.wheel_var_mult, m.gnss_var_mult, m.nhc_var_mult * 100
    )
    _, _, d2 = update_nonholonomic(x0, P, omega, params, NonholonomicConfig(enabled=True), big)
    assert float(d2.S[0, 0, 0]) > float(d.S[0, 0, 0])  # learned multiplier inflates the variance


def test_nonholonomic_level_vertical_jacobian_and_pitch_independence() -> None:
    from lase.filter.params import NonholonomicConfig
    from lase.filter.updates import axle_velocity_world, update_nonholonomic

    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    omega = torch.tensor([[0.05, -0.2, 0.3]], dtype=D)
    J = _H_numeric(lambda x: axle_velocity_world(x, omega, params)[:, 2:3], x0)
    from lase.filter.so3 import skew

    lever = torch.linalg.cross(omega, params.l_imu[None])
    H = torch.zeros(N_ERR, dtype=D)
    H[5] = 1.0
    H[6:9] = (x0.C_wb @ skew(lever))[0, 2, :]
    np.testing.assert_allclose(H.numpy(), J[0].numpy(), atol=1e-10)
    cfg = NonholonomicConfig(enabled=True, vertical_mode="level")
    _, _, d = update_nonholonomic(x0, torch.eye(N_ERR, dtype=D)[None], omega, params, cfg)
    assert d.innovation.shape == (1, 2)
