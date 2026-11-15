import numpy as np
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
    )


def _H_numeric(fun, x0: NavState) -> torch.Tensor:  # type: ignore[no-untyped-def]
    return torch.autograd.functional.jacobian(
        lambda dx: fun(x0.inject(dx[None]))[0], torch.zeros(N_ERR, dtype=D)
    )


def test_wheel_jacobian_matches_autograd() -> None:
    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    omega = torch.tensor([[0.05, -0.02, 0.3]], dtype=D)
    J = _H_numeric(lambda x: params.wheel_scale * wheel_prediction(x, omega, params)[:, None], x0)
    # Recover H by an update with tiny P and a known innovation: use the function directly instead
    C = x0.C_wb
    v_b = (C.transpose(-1, -2) @ x0.v_w[..., None])[..., 0]
    from lase.filter.so3 import skew

    H = torch.zeros(1, N_ERR, dtype=D)
    H[0, 3:6] = params.wheel_scale * C[0, :, 0]
    H[0, 6:9] = params.wheel_scale * skew(v_b)[0, 0, :]
    np.testing.assert_allclose(H.numpy(), J.numpy(), atol=1e-10)


def test_gnss_jacobian_matches_autograd() -> None:
    params = FilterParams.build(load_calibration(), FilterConfig())
    x0 = _state()
    J = _H_numeric(lambda x: gnss_prediction(x, params), x0)
    from lase.filter.so3 import skew

    d = params.l_gps - params.l_imu
    H = torch.zeros(2, N_ERR, dtype=D)
    H[:, 0:3] = torch.eye(3, dtype=D)[:2]
    H[:, 6:9] = -(x0.C_wb[0] @ skew(d))[:2]
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
