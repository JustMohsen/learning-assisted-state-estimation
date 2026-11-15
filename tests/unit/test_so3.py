import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lase.filter.so3 import (
    matrix_to_quat,
    quat_exp,
    quat_mul,
    quat_to_matrix,
    skew,
    so3_exp,
    so3_log,
    yaw_of,
)

torch.set_default_dtype(torch.float64)


def _rotvecs(n: int = 200) -> torch.Tensor:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0, 3.0, (n, 1))
    v[:5] *= 1e-9  # tiny angles
    return torch.as_tensor(v)


def test_exp_matches_scipy_and_log_inverts() -> None:
    phi = _rotvecs()
    C = so3_exp(phi)
    np.testing.assert_allclose(C.numpy(), Rotation.from_rotvec(phi.numpy()).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(so3_log(C).numpy(), phi.numpy(), atol=1e-8)


def test_quaternion_consistency() -> None:
    phi = _rotvecs()
    q = quat_exp(phi)
    np.testing.assert_allclose(quat_to_matrix(q).numpy(), so3_exp(phi).numpy(), atol=1e-12)
    q2 = matrix_to_quat(quat_to_matrix(q))
    np.testing.assert_allclose(quat_to_matrix(q2).numpy(), quat_to_matrix(q).numpy(), atol=1e-12)
    # composition: Exp(a) Exp(b) == R(q_a ⊗ q_b)
    a, b = phi[:100], phi[100:]
    np.testing.assert_allclose(
        quat_to_matrix(quat_mul(quat_exp(a), quat_exp(b))).numpy(),
        (so3_exp(a) @ so3_exp(b)).numpy(),
        atol=1e-12,
    )


def test_skew_cross_and_yaw() -> None:
    a, b = torch.randn(10, 3), torch.randn(10, 3)
    torch.testing.assert_close(skew(a) @ b[..., None], torch.linalg.cross(a, b)[..., None])
    C = so3_exp(torch.tensor([[0.0, 0.0, 1.2]]))
    assert abs(float(yaw_of(C)[0]) - 1.2) < 1e-12


def test_exp_gradient_finite_at_zero() -> None:
    phi = torch.zeros(1, 3, requires_grad=True)
    so3_exp(phi).sum().backward()
    assert phi.grad is not None
    assert torch.isfinite(phi.grad).all()


def test_right_jacobian_definition() -> None:
    from lase.filter.so3 import so3_right_jacobian

    phi = _rotvecs(50)[5:]
    d = 1e-7 * torch.randn(45, 3)
    lhs = so3_exp(phi + d)
    rhs = so3_exp(phi) @ so3_exp((so3_right_jacobian(phi) @ d[..., None])[..., 0])
    np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), atol=1e-12)
