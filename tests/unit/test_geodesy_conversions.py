import numpy as np

from lase.data.calibration import load_calibration
from lase.data.conversions import (
    fog_increments,
    interp_at,
    rate_on_grid,
    unwrap_fog,
    wheel_forward_speed,
    wheel_yaw_rate,
)
from lase.data.geodesy import LocalProjection, geodetic_to_ned_wgs84


def _proj() -> LocalProjection:
    return LocalProjection.from_config(load_calibration().local_frame)


def test_origin_maps_to_zero_and_axes() -> None:
    p = _proj()
    np.testing.assert_allclose(p.to_local(p.lat0, p.lon0, p.alt0), [0, 0, 0], atol=1e-9)
    north = p.to_local(p.lat0 + 1e-5, p.lon0, p.alt0)
    east = p.to_local(p.lat0, p.lon0 + 1e-5, p.alt0)
    up = p.to_local(p.lat0, p.lon0, p.alt0 + 10)
    assert north[0] > 60
    assert abs(north[1]) < 1e-9
    assert east[1] > 45
    assert abs(east[0]) < 1e-9
    np.testing.assert_allclose(up, [0, 0, -10])


def test_roundtrip_and_agreement_with_wgs84() -> None:
    p = _proj()
    rng = np.random.default_rng(1)
    ned = rng.uniform([-1500, -1500, -30], [1500, 1500, 30], size=(200, 3))
    lat, lon, alt = p.to_geodetic(ned)
    np.testing.assert_allclose(p.to_local(lat, lon, alt), ned, atol=1e-6)
    exact = geodetic_to_ned_wgs84(lat, lon, alt, p.lat0, p.lon0, p.alt0)
    # Linearisation vs exact WGS-84 within a 1.5 km campus: sub-metre horizontally.
    assert np.max(np.abs(exact[:, :2] - ned[:, :2])) < 1.0


def test_fog_unwrap_and_increments() -> None:
    wrapped = np.mod(np.linspace(0, 4 * np.pi, 50), 2 * np.pi)
    un = unwrap_fog(wrapped)
    np.testing.assert_allclose(un, np.linspace(0, 4 * np.pi, 50), atol=1e-12)
    inc = fog_increments(un)
    assert inc[0] == 0
    np.testing.assert_allclose(inc[1:].sum(), un[-1] - un[0])


def test_interp_and_rate() -> None:
    t = np.arange(0, 2_000_001, 10_000, dtype=np.int64)  # 2 s at 100 Hz
    angle = 0.5 * t * 1e-6  # 0.5 rad/s
    grid = np.arange(100_000, 1_900_001, 100_000, dtype=np.int64)
    np.testing.assert_allclose(rate_on_grid(t, angle, grid), 0.5, atol=1e-9)
    out = interp_at(t, angle, np.array([-5, 1_000_000], dtype=np.int64))
    assert np.isnan(out[0])
    assert np.isclose(out[1], 0.5)


def test_wheel_kinematics() -> None:
    np.testing.assert_allclose(wheel_forward_speed(np.array([1.0]), np.array([3.0])), [2.0])
    # left faster -> right turn -> positive yaw (z down)
    assert wheel_yaw_rate(np.array([1.2]), np.array([1.0]), 0.5)[0] > 0
