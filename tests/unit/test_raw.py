from pathlib import Path

import numpy as np

from lase.data.raw import (
    RowFlag,
    load_euler,
    load_fog,
    load_gps,
    load_imu,
    load_odometry,
    load_rtk_error,
    load_wheels,
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_imu_parse_and_flags(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "ms25.csv",
        "100,0.1,0.2,0.3,0,0,-9.8,0.01,0.02,0.03\n"
        "200,0.1,0.2,0.3,0,0,-9.8,0.01,0.02,0.03\n"
        "200,0.1,0.2,0.3,0,0,-9.8,0.01,0.02,0.03\n"  # duplicate time
        "150,0.1,0.2,0.3,0,0,-9.8,0.01,0.02,0.03\n"  # non-monotonic
        "300,0.1,0.2,0.3,0,0,-99,0.01,0.02,0.03\n"  # out of range
        "400,0.1,0.2,0.3,nan,0,-9.8,0.01,0.02,0.03\n",  # non-finite
    )
    r = load_imu(p)
    assert len(r) == 6
    assert r.t_us.dtype == np.int64
    np.testing.assert_allclose(r.acc[0], [0, 0, -9.8])
    np.testing.assert_allclose(r.gyr[0], [0.01, 0.02, 0.03])
    assert r.flags[2] & RowFlag.DUPLICATE_TIME
    assert r.flags[3] & RowFlag.NON_MONOTONIC
    assert r.flags[4] & RowFlag.OUT_OF_RANGE
    assert r.flags[5] & RowFlag.NON_FINITE
    assert r.valid.tolist() == [True, True, True, False, False, False]


def test_fog_and_wheels(tmp_path: Path) -> None:
    fog = load_fog(_write(tmp_path, "kvh.csv", "1,6.2\n2,6.25\n"))
    np.testing.assert_allclose(fog.angle, [6.2, 6.25])
    w = load_wheels(_write(tmp_path, "wheels.csv", "1,-0.01,0.02\n2,9.0,1.0\n"))
    np.testing.assert_allclose(w.left, [-0.01, 9.0])
    assert w.valid.tolist() == [True, False]


def test_gps_flags(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "gps.csv",
        "1,3,0,0.7381677,-1.4609866,275.2,7.4,0.3\n"
        "2,3,0,0.7381677,-1.4609866,275.2,7.4,0.3\n"  # held
        "3,1,0,0.7381678,-1.4609866,275.2,nan,nan\n"  # no fix
        "4,3,0,nan,nan,nan,nan,nan\n",  # non-finite
    )
    g = load_gps(p)
    assert g.mode.tolist() == [3, 3, 1, 3]
    assert g.flags[1] & RowFlag.GPS_HELD
    assert g.flags[2] & RowFlag.GPS_NO_FIX
    assert g.flags[3] & RowFlag.NON_FINITE
    assert g.valid.tolist() == [True, True, False, False]


def test_diagnostic_loaders(tmp_path: Path) -> None:
    e = load_euler(_write(tmp_path, "e.csv", "1,0.1,0.2,0.3\n"))
    np.testing.assert_allclose(e.rpy, [[0.1, 0.2, 0.3]])
    o = load_odometry(_write(tmp_path, "o.csv", "1,1,2,3,0.1,0.2,0.3\n"))
    assert o.pose.shape == (1, 6)
    r = load_rtk_error(_write(tmp_path, "r.csv", "1, 1.5\n"))
    np.testing.assert_allclose(r.err, [1.5])
