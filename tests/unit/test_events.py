import dataclasses
import datetime as dt

import numpy as np

from lase.data.events import SensorId, build_event_index, iter_events
from lase.data.raw import FogRecord, GpsRecord, ImuRecord, WheelRecord
from lase.data.session import SensorStreams


def _streams() -> SensorStreams:
    def z(n: int, k: int = 0) -> np.ndarray:
        return np.zeros((n, k)) if k else np.zeros(n)

    def f(n: int) -> np.ndarray:
        return np.zeros(n, dtype=np.uint16)

    imu_t = np.array([10, 20, 30], dtype=np.int64)
    fog_t = np.array([10, 15, 30], dtype=np.int64)
    wh_t = np.array([20], dtype=np.int64)
    gps_t = np.array([10, 25], dtype=np.int64)
    gps_flags = f(2)
    gps_flags[1] = 16  # GPS_NO_FIX -> invalid
    return SensorStreams(
        date=dt.date(2013, 1, 10),
        imu=ImuRecord(t_us=imu_t, flags=f(3), mag=z(3, 3), acc=z(3, 3), gyr=z(3, 3)),
        fog=FogRecord(t_us=fog_t, flags=f(3), angle=z(3)),
        wheels=WheelRecord(t_us=wh_t, flags=f(1), left=z(1), right=z(1)),
        gps=GpsRecord(
            t_us=gps_t,
            flags=gps_flags,
            mode=np.array([3, 1]),
            num_sats=np.zeros(2, dtype=np.int64),
            lat=z(2),
            lon=z(2),
            alt=z(2),
            track=z(2),
            speed=z(2),
        ),
        fog_angle=np.array([0.0, 0.1, 0.2]),
        gnss_ned=np.arange(6, dtype=float).reshape(2, 3),
    )


def test_order_and_tie_priority() -> None:
    idx = build_event_index(_streams())
    assert np.all(np.diff(idx.t_us) >= 0)
    # at t=10: IMU, FOG, GNSS in priority order
    at10 = [SensorId(int(s)) for s, t in zip(idx.sensor, idx.t_us, strict=True) if t == 10]
    assert at10 == [SensorId.IMU, SensorId.FOG, SensorId.GNSS]
    at20 = [SensorId(int(s)) for s, t in zip(idx.sensor, idx.t_us, strict=True) if t == 20]
    assert at20 == [SensorId.IMU, SensorId.WHEEL]


def test_window_half_open() -> None:
    w = build_event_index(_streams()).window(15, 30)
    assert w.t_us.tolist() == [15, 20, 20, 25]


def test_payload_and_validity() -> None:
    ev = list(iter_events(_streams()))
    gnss = [e for e in ev if e.sensor is SensorId.GNSS]
    np.testing.assert_allclose(gnss[0].data, [0, 1, 2, 3])
    assert gnss[0].valid
    assert not gnss[1].valid
    assert len(ev) == 9


def test_inputs_contain_no_reference_fields() -> None:
    names = {f.name for f in dataclasses.fields(SensorStreams)}
    assert names == {"date", "imu", "fog", "wheels", "gps", "fog_angle", "gnss_ned"}
