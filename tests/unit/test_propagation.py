import numpy as np
import torch

from lase.data.calibration import load_calibration
from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.propagation import ImuInterval, boxminus, propagate
from lase.filter.so3 import matrix_to_quat, so3_exp
from lase.filter.state import N_ERR, NavState
from lase.sim.strapdown import strapdown

D = torch.float64


def _params(imu_aligned: bool = True) -> FilterParams:
    return FilterParams.build(load_calibration(), FilterConfig())


def _state(seed: int = 0) -> NavState:
    g = torch.Generator().manual_seed(seed)
    C = so3_exp(0.3 * torch.randn(1, 3, generator=g, dtype=D))
    return NavState(
        p_w=torch.randn(1, 3, generator=g, dtype=D),
        v_w=torch.randn(1, 3, generator=g, dtype=D),
        q_wb=matrix_to_quat(C),
        b_acc=0.05 * torch.randn(1, 3, generator=g, dtype=D),
        b_gyr=0.01 * torch.randn(1, 3, generator=g, dtype=D),
        b_fog=1e-4 * torch.randn(1, 1, generator=g, dtype=D),
        wheel_scale=torch.ones(1, 1, dtype=D),
        gps_err=torch.tensor([[0.7, -0.4]], dtype=D),
        grade=torch.tensor([[0.03, -0.02]], dtype=D),
    )


def _interval(seed: int = 1, fog_valid: float = 1.0) -> ImuInterval:
    g = torch.Generator().manual_seed(seed)
    acc = torch.randn(2, 3, generator=g, dtype=D) + torch.tensor([0.0, 0.0, -9.8], dtype=D)
    gyr = 0.5 * torch.randn(2, 3, generator=g, dtype=D)
    return ImuInterval(
        dt=torch.tensor([0.02], dtype=D),
        acc0=acc[:1],
        acc1=acc[1:],
        gyr0=gyr[:1],
        gyr1=gyr[1:],
        fog_dangle=torch.tensor([-0.006], dtype=D),
        fog_valid=torch.tensor([fog_valid], dtype=D),
    )


def test_mean_matches_strapdown_without_fog() -> None:
    rng = np.random.default_rng(0)
    n = 300
    t = np.cumsum(np.r_[0, rng.uniform(0.015, 0.03, n - 1)] * 1e6).astype(np.int64)
    acc = rng.normal(0, 1, (n, 3)) + np.array([0.0, 0.0, -9.8])
    gyr = rng.normal(0, 0.3, (n, 3))
    p_ref, v_ref, C_ref = strapdown(t, acc, gyr, np.zeros(3), np.array([1.0, 0, 0]), np.eye(3))
    params = _params()
    params = FilterParams(
        l_imu=params.l_imu,
        l_gps=params.l_gps,
        C_b_imu=torch.eye(3, dtype=D),
        fog_sign=-1.0,
        wheel_scale=1.0,
        gravity_w=params.gravity_w,
        earth_rate_w=torch.zeros(3, dtype=D),  # the reference strapdown ignores Earth rate
    )
    s = NavState(
        torch.zeros(1, 3, dtype=D),
        torch.tensor([[1.0, 0, 0]], dtype=D),
        torch.tensor([[1.0, 0, 0, 0]], dtype=D),
        torch.zeros(1, 3, dtype=D),
        torch.zeros(1, 3, dtype=D),
        torch.zeros(1, 1, dtype=D),
        torch.ones(1, 1, dtype=D),
        torch.zeros(1, 2, dtype=D),
        torch.zeros(1, 2, dtype=D),
    )
    P = torch.eye(N_ERR, dtype=D)[None]
    cfg = FilterConfig()
    A, G = torch.as_tensor(acc), torch.as_tensor(gyr)
    for k in range(n - 1):
        iv = ImuInterval(
            dt=torch.tensor([(t[k + 1] - t[k]) * 1e-6], dtype=D),
            acc0=A[k : k + 1],
            acc1=A[k + 1 : k + 2],
            gyr0=G[k : k + 1],
            gyr1=G[k + 1 : k + 2],
            fog_dangle=torch.zeros(1, dtype=D),
            fog_valid=torch.zeros(1, dtype=D),
        )
        s, P, _ = propagate(s, P, iv, params, cfg.noise, HeadOutputs.neutral(1))
    np.testing.assert_allclose(s.p_w[0].numpy(), p_ref[-1], atol=1e-9)
    np.testing.assert_allclose(s.v_w[0].numpy(), v_ref[-1], atol=1e-9)
    np.testing.assert_allclose(s.C_wb[0].numpy(), C_ref[-1], atol=1e-9)


def _check_jacobian(fog_valid: float) -> None:
    params, cfg = _params(), FilterConfig()
    x0, iv = _state(), _interval(fog_valid=fog_valid)
    P = torch.eye(N_ERR, dtype=D)[None]
    learned = HeadOutputs.neutral(1)
    base, _, diag = propagate(x0, P, iv, params, cfg.noise, learned)

    def f(dx: torch.Tensor) -> torch.Tensor:
        out, _, _ = propagate(x0.inject(dx[None]), P, iv, params, cfg.noise, learned)
        return boxminus(out, base)[0]

    J = torch.autograd.functional.jacobian(f, torch.zeros(N_ERR, dtype=D))
    np.testing.assert_allclose(diag.F[0].numpy(), J.numpy(), atol=1e-8, rtol=1e-6)


def test_covariance_jacobian_matches_autograd_with_fog() -> None:
    _check_jacobian(1.0)


def test_covariance_jacobian_matches_autograd_without_fog() -> None:
    _check_jacobian(0.0)


def test_jacobian_with_gps_error_state_enabled() -> None:
    from lase.filter.params import FilterConfig as FC

    base = FC()
    cfg = base.model_copy(update={"gnss": base.gnss.model_copy(update={"error_state": True})})
    params, x0, iv = FilterParams.build(load_calibration(), cfg), _state(), _interval()
    P = torch.eye(N_ERR, dtype=D)[None]
    learned = HeadOutputs.neutral(1)
    base_out, _, diag = propagate(x0, P, iv, params, cfg.noise, learned)

    def f(dx: torch.Tensor) -> torch.Tensor:
        out, _, _ = propagate(x0.inject(dx[None]), P, iv, params, cfg.noise, learned)
        return boxminus(out, base_out)[0]

    J = torch.autograd.functional.jacobian(f, torch.zeros(N_ERR, dtype=D))
    np.testing.assert_allclose(diag.F[0].numpy(), J.numpy(), atol=1e-8, rtol=1e-6)
    phi = float(np.exp(-0.02 / cfg.gnss.error_gm_tau_s))
    np.testing.assert_allclose(diag.F[0, 17:19, 17:19].numpy(), phi * np.eye(2), atol=1e-12)
