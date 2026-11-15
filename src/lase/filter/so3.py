"""SO(3) and quaternion operations in PyTorch (batched over leading dims, float64).

Quaternions are Hamilton, scalar first ``[w, x, y, z]`` (CONVENTIONS section 4).
Small-angle branches use Taylor expansions so values and gradients stay finite at 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def skew(v: Tensor) -> Tensor:
    """``[v]x`` with ``skew(a) @ b == cross(a, b)``. Shape (..., 3) -> (..., 3, 3)."""
    z = torch.zeros_like(v[..., 0])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([z, -w, y], dim=-1),
            torch.stack([w, z, -x], dim=-1),
            torch.stack([-y, x, z], dim=-1),
        ],
        dim=-2,
    )


def _sinc_terms(theta2: Tensor) -> tuple[Tensor, Tensor]:
    """A = sin(t)/t and B = (1 - cos t)/t^2, safe at t = 0."""
    small = theta2 < _EPS
    t2 = torch.where(small, torch.ones_like(theta2), theta2)
    t = torch.sqrt(t2)
    a = torch.where(small, 1 - theta2 / 6 + theta2**2 / 120, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2**2 / 720, (1 - torch.cos(t)) / t2)
    return a, b


def so3_exp(phi: Tensor) -> Tensor:
    """Rotation matrix ``Exp(phi)`` (Rodrigues). Shape (..., 3) -> (..., 3, 3)."""
    theta2 = (phi * phi).sum(-1)
    a, b = _sinc_terms(theta2)
    K = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)


def so3_log(C: Tensor) -> Tensor:
    """Rotation vector ``Log(C)`` for angles below pi. Shape (..., 3, 3) -> (..., 3)."""
    cos = ((C.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    theta = torch.acos(cos)
    small = theta < 1e-6
    factor = torch.where(
        small,
        0.5 + theta**2 / 12,
        theta / (2 * torch.sin(torch.where(small, torch.ones_like(theta), theta))),
    )
    v = torch.stack(
        [C[..., 2, 1] - C[..., 1, 2], C[..., 0, 2] - C[..., 2, 0], C[..., 1, 0] - C[..., 0, 1]],
        dim=-1,
    )
    return factor[..., None] * v


def quat_mul(q: Tensor, r: Tensor) -> Tensor:
    """Hamilton product ``q ⊗ r``. Shape (..., 4)."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_exp(phi: Tensor) -> Tensor:
    """Unit quaternion of rotation vector ``phi``. Shape (..., 3) -> (..., 4)."""
    half2 = (phi * phi).sum(-1) / 4
    small = half2 < _EPS
    h2 = torch.where(small, torch.ones_like(half2), half2)
    h = torch.sqrt(h2)
    w = torch.where(small, 1 - half2 / 2 + half2**2 / 24, torch.cos(h))
    s = torch.where(small, 0.5 - half2 / 12 + half2**2 / 240, torch.sin(h) / (2 * h))
    return torch.cat([w[..., None], s[..., None] * phi], dim=-1)


def quat_normalize(q: Tensor) -> Tensor:
    """Normalise to unit norm with non-negative scalar part."""
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrix of a unit quaternion. Shape (..., 4) -> (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def matrix_to_quat(C: Tensor) -> Tensor:
    """Unit quaternion of a rotation matrix (Shepperd-style, branch on largest component)."""
    m = C
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = torch.stack(
        [
            torch.stack(
                [
                    1 + tr,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    1 + 2 * m[..., 0, 0] - tr,
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 0, 1] + m[..., 1, 0],
                    1 + 2 * m[..., 1, 1] - tr,
                    m[..., 1, 2] + m[..., 2, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    1 + 2 * m[..., 2, 2] - tr,
                ],
                -1,
            ),
        ],
        dim=-2,
    )
    diag = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    k = diag.argmax(-1)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return quat_normalize(q)


def yaw_of(C: Tensor) -> Tensor:
    """Yaw of ``C_wb`` under the NCLT ZYX convention (rad, wrapped)."""
    return torch.atan2(C[..., 1, 0], C[..., 0, 0])


def wrap(a: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def so3_right_jacobian(phi: Tensor) -> Tensor:
    """Right Jacobian ``J_r(phi)``: ``Exp(phi + d) ≈ Exp(phi) Exp(J_r(phi) d)``. (..., 3, 3)."""
    theta2 = (phi * phi).sum(-1)
    small = theta2 < _EPS
    t2 = torch.where(small, torch.ones_like(theta2), theta2)
    t = torch.sqrt(t2)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(t)) / t2)
    c = torch.where(small, 1 / 6 - theta2 / 120, (t - torch.sin(t)) / (t2 * t))
    K = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return eye - b[..., None, None] * K + c[..., None, None] * (K @ K)
