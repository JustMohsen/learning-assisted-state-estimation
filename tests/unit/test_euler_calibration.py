import numpy as np
import pytest

from lase.data.calibration import load_calibration
from lase.geometry.euler import euler_zyx_to_matrix, matrix_to_euler_zyx, rot_z, wrap_angle


def test_rot_z_maps_north_to_east_for_positive_yaw() -> None:
    # NED: positive yaw about Down turns North (x) toward East (y).
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1.0, 0, 0], [0, 1.0, 0], atol=1e-12)


def test_euler_roundtrip() -> None:
    rng = np.random.default_rng(0)
    rpy = rng.uniform([-1, -1.4, -3], [1, 1.4, 3], size=(100, 3))
    C = euler_zyx_to_matrix(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    np.testing.assert_allclose(
        C @ C.transpose(0, 2, 1), np.broadcast_to(np.eye(3), C.shape), atol=1e-12
    )
    back = np.stack(matrix_to_euler_zyx(C), axis=-1)
    np.testing.assert_allclose(back, rpy, atol=1e-10)


def test_wrap_angle() -> None:
    np.testing.assert_allclose(
        wrap_angle([np.pi, -np.pi, 3 * np.pi, 0.1]), [np.pi, np.pi, np.pi, 0.1]
    )


def test_calibration_loads_with_published_mounts() -> None:
    cal = load_calibration()
    assert cal.local_frame.lat_origin_deg == pytest.approx(42.293227)
    np.testing.assert_allclose(cal.sensors.imu.l_b, [-0.11, -0.18, -0.71])
    np.testing.assert_allclose(cal.sensors.imu.C_bs, np.eye(3))
    np.testing.assert_allclose(cal.sensors.gps.C_bs, np.eye(3))
    assert cal.wheels.track_width_m is None
