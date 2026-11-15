import numpy as np

from lase.geometry.constants import GRAVITY_MPS2
from lase.geometry.euler import euler_zyx_to_matrix
from lase.sim.trajectory import euler_rates_to_body


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def test_euler_rates_match_rotation_derivative() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        rpy = rng.uniform([-0.5, -0.5, -3], [0.5, 0.5, 3])
        rate = rng.uniform(-1, 1, 3)
        h = 1e-6
        C0 = euler_zyx_to_matrix(*rpy)
        C1 = euler_zyx_to_matrix(*(rpy + h * rate))
        C_dot = (C1 - C0) / h
        w = euler_rates_to_body(rpy[None], rate[None])[0]
        np.testing.assert_allclose(C_dot, C0 @ _skew(w), atol=1e-5)  # C' = C [w]x


def test_gravity_value() -> None:
    assert 9.80 < GRAVITY_MPS2 < 9.805
