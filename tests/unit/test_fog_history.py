import datetime as dt

import numpy as np

from lase.data.raw import FogRecord, GpsRecord, ImuRecord, WheelRecord
from lase.data.session import SensorStreams
from lase.filter.runner import _FogHistory


def _streams(t: np.ndarray, a: np.ndarray) -> SensorStreams:
    z1, f1, t1 = np.zeros(1), np.zeros(1, dtype=np.uint16), np.zeros(1, dtype=np.int64)
    return SensorStreams(
        date=dt.date(2000, 1, 1),
        imu=ImuRecord(t1, f1, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))),
        fog=FogRecord(t, np.zeros(len(t), dtype=np.uint16), a),
        wheels=WheelRecord(t1, f1, z1, z1),
        gps=GpsRecord(t1, f1, np.ones(1, dtype=np.int64), t1, z1, z1, z1, z1, z1),
        fog_angle=a,
        gnss_ned=np.zeros((1, 3)),
    )


def test_interpolation_and_causality() -> None:
    t = np.array([0, 10_000, 20_000, 30_000], dtype=np.int64)
    a = np.array([0.0, 0.1, 0.2, 0.3])
    h = _FogHistory(_streams(t, a))
    assert abs(h.angle_at(15_000, t_now=30_000) - 0.15) < 1e-12  # type: ignore[operator]
    # at t_now=20_000 the sample at 30_000 is not yet available: extrapolate
    assert abs(h.angle_at(25_000, t_now=20_000) - 0.25) < 1e-12  # type: ignore[operator]
    assert h.angle_at(200_000, t_now=30_000) is None  # beyond the extrapolation limit


def test_extrapolation_robust_to_nearly_coincident_samples() -> None:
    # rate 1 rad/s, but the last two samples are 0.1 ms apart with a 1e-4 rad angle jitter
    t = np.array([0, 10_000, 20_000, 20_100], dtype=np.int64)
    a = np.array([0.0, 0.01, 0.02, 0.0201 + 1e-4])
    h = _FogHistory(_streams(t, a))
    v = h.angle_at(30_000, t_now=20_100)
    assert v is not None
    assert abs(v - 0.03) < 2e-3  # a two-sample rate would give an error of ~10 mrad
