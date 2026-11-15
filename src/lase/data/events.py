"""Causal event stream over estimator inputs.

All estimator-input samples are merged into one time-ordered index. Ties at equal
timestamps follow the fixed priority of :class:`SensorId` (propagation inputs before
updates). Only :class:`~lase.data.session.SensorStreams` can be indexed, so reference
and diagnostic data cannot enter the stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from lase.data.raw import F64, I64, B
from lase.data.session import SensorStreams


class SensorId(IntEnum):
    """Estimator-input sensors; the value is the tie-break priority (lower first)."""

    IMU = 0
    FOG = 1
    WHEEL = 2
    GNSS = 3


@dataclass(frozen=True, slots=True)
class Event:
    """One sensor sample. ``data`` is in SI units (see :func:`payload`)."""

    t_us: int
    sensor: SensorId
    row: int
    valid: bool
    data: F64


@dataclass(frozen=True, slots=True)
class EventIndex:
    """Time-ordered index of all input samples: (time, sensor, row in that sensor's record)."""

    t_us: I64
    sensor: npt.NDArray[np.int8]
    row: I64
    valid: B

    def __len__(self) -> int:
        return len(self.t_us)

    def window(self, t0_us: int, t1_us: int) -> EventIndex:
        """Events with ``t0_us <= t < t1_us``."""
        a = int(np.searchsorted(self.t_us, t0_us, side="left"))
        b = int(np.searchsorted(self.t_us, t1_us, side="left"))
        return EventIndex(self.t_us[a:b], self.sensor[a:b], self.row[a:b], self.valid[a:b])


def build_event_index(streams: SensorStreams) -> EventIndex:
    """Merge all input records into one stably time-sorted index."""
    parts = [
        (SensorId.IMU, streams.imu.t_us, streams.imu.valid),
        (SensorId.FOG, streams.fog.t_us, streams.fog.valid),
        (SensorId.WHEEL, streams.wheels.t_us, streams.wheels.valid),
        (SensorId.GNSS, streams.gps.t_us, streams.gps.valid),
    ]
    t = np.concatenate([p[1] for p in parts])
    sensor = np.concatenate([np.full(len(p[1]), int(p[0]), dtype=np.int8) for p in parts])
    row = np.concatenate([np.arange(len(p[1]), dtype=np.int64) for p in parts])
    valid = np.concatenate([p[2] for p in parts])
    order = np.lexsort((row, sensor, t))  # by time, then sensor priority, then row
    return EventIndex(t[order], sensor[order], row[order], valid[order])


def payload(streams: SensorStreams, sensor: SensorId, row: int) -> F64:
    """Data of one sample.

    - IMU: ``[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]`` (m/s^2, rad/s; IMU axes)
    - FOG: ``[unwrapped_angle]`` (rad)
    - WHEEL: ``[left, right]`` (m/s)
    - GNSS: ``[north, east, down, fix_mode]`` (m; antenna position in local NED)
    """
    if sensor is SensorId.IMU:
        return np.concatenate([streams.imu.acc[row], streams.imu.gyr[row]])
    if sensor is SensorId.FOG:
        return np.array([streams.fog_angle[row]])
    if sensor is SensorId.WHEEL:
        return np.array([streams.wheels.left[row], streams.wheels.right[row]])
    return np.append(streams.gnss_ned[row], float(streams.gps.mode[row]))


def iter_events(streams: SensorStreams, index: EventIndex | None = None) -> Iterator[Event]:
    """Iterate events in causal order (convenience; vectorised code should use the index)."""
    idx = index if index is not None else build_event_index(streams)
    for t, s, r, v in zip(idx.t_us, idx.sensor, idx.row, idx.valid, strict=True):
        sid = SensorId(int(s))
        yield Event(int(t), sid, int(r), bool(v), payload(streams, sid, int(r)))
