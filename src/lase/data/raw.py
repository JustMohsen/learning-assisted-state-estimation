"""Raw loaders for NCLT sensor CSVs.

Each loader parses one file with an explicit column schema and returns an immutable
record of NumPy arrays. Timestamps stay ``int64`` microseconds (``t_us``). Rows are
**never dropped**: problems are reported in a per-row ``flags`` bitmask, and
``valid`` marks rows without any invalidating flag. Units are those of the release
(SI, angles in radians), checked during the Phase 1 audit.

Files and roles:

- estimator inputs: ``ms25.csv`` (IMU), ``kvh.csv`` (FOG), ``wheels.csv``, ``gps.csv``
- diagnostics only: ``ms25_euler.csv``, ``gps_rtk.csv``, ``gps_rtk_err.csv``,
  ``odometry_mu*.csv``, ``odometry_cov*.csv``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from lase.utils.config import Config

LOADER_VERSION = 1
"""Bump when parsing or flagging changes, to invalidate caches."""

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
U16 = npt.NDArray[np.uint16]
B = npt.NDArray[np.bool_]


class RowFlag(IntFlag):
    """Per-row data-quality flags."""

    NONE = 0
    NON_MONOTONIC = 1  # timestamp earlier than the previous row's
    DUPLICATE_TIME = 2  # timestamp equal to the previous row's
    NON_FINITE = 4  # NaN/inf in a required value
    OUT_OF_RANGE = 8  # value outside physical plausibility limits
    GPS_NO_FIX = 16  # receiver reports no position fix
    GPS_HELD = 32  # position identical to the previous fix (held or stationary)


INVALID_MASK = (
    RowFlag.NON_MONOTONIC | RowFlag.NON_FINITE | RowFlag.OUT_OF_RANGE | RowFlag.GPS_NO_FIX
)
"""Flags that make a row invalid. DUPLICATE_TIME and GPS_HELD are informational."""


class RawLimits(Config):
    """Plausibility limits for flagging (generous; physical bounds, not tuning)."""

    max_specific_force_mps2: float = 60.0
    max_angular_rate_radps: float = 10.0
    max_wheel_speed_mps: float = 5.0
    min_gps_fix_mode: int = 2


@dataclass(frozen=True, slots=True)
class _Base:
    t_us: I64
    flags: U16

    @property
    def valid(self) -> B:
        """Rows without any invalidating flag."""
        valid: B = (self.flags & np.uint16(INVALID_MASK)) == 0
        return valid

    def __len__(self) -> int:
        return len(self.t_us)


@dataclass(frozen=True, slots=True)
class ImuRecord(_Base):
    """Microstrain MS25: magnetometer (gauss), specific force (m/s^2), angular rate (rad/s), sensor axes."""

    mag: F64  # (N, 3)
    acc: F64  # (N, 3)
    gyr: F64  # (N, 3)


@dataclass(frozen=True, slots=True)
class FogRecord(_Base):
    """KVH DSP-3000: integrated heading angle (rad), arbitrary reference, possibly wrapped."""

    angle: F64  # (N,)


@dataclass(frozen=True, slots=True)
class WheelRecord(_Base):
    """Left and right wheel speeds (m/s)."""

    left: F64  # (N,)
    right: F64  # (N,)


@dataclass(frozen=True, slots=True)
class GpsRecord(_Base):
    """Receiver-processed GPS fixes. Latitude/longitude in radians, altitude in metres."""

    mode: I64  # fix mode
    num_sats: I64
    lat: F64
    lon: F64
    alt: F64
    track: F64  # semantics/units unverified; not used as an observation
    speed: F64  # not used as an observation


@dataclass(frozen=True, slots=True)
class EulerRecord(_Base):
    """Vendor IMU attitude (roll, pitch, heading; rad). Diagnostic only."""

    rpy: F64  # (N, 3)


@dataclass(frozen=True, slots=True)
class OdometryRecord(_Base):
    """Supplied odometry pose (x, y, z, roll, pitch, yaw). Diagnostic only: fuses IMU/wheels/FOG."""

    pose: F64  # (N, 6)


@dataclass(frozen=True, slots=True)
class RtkErrorRecord(_Base):
    """RTK error estimate (m) as released. Diagnostic only."""

    err: F64  # (N,)


def _read(path: Path, ncols: int) -> tuple[I64, F64]:
    """Read a headerless numeric CSV: int64 timestamps and float64 value columns."""
    df = pd.read_csv(
        path,
        header=None,
        skipinitialspace=True,
        dtype={0: np.int64, **dict.fromkeys(range(1, ncols), np.float64)},
        names=list(range(ncols)),
    )
    t = df[0].to_numpy(np.int64)
    v = df.iloc[:, 1:].to_numpy(np.float64)
    return t, v


def _time_flags(t_us: I64) -> U16:
    flags = np.zeros(len(t_us), dtype=np.uint16)
    if len(t_us) > 1:
        dt = np.diff(t_us)
        flags[1:][dt < 0] |= np.uint16(RowFlag.NON_MONOTONIC)
        flags[1:][dt == 0] |= np.uint16(RowFlag.DUPLICATE_TIME)
    return flags


def _flag(flags: U16, mask: B, flag: RowFlag) -> None:
    flags[mask] |= np.uint16(flag)


def load_imu(path: Path, limits: RawLimits | None = None) -> ImuRecord:
    """Load ``ms25.csv``: utime, mag xyz, acc xyz, gyro xyz."""
    lim = limits or RawLimits()
    t, v = _read(path, 10)
    mag, acc, gyr = v[:, 0:3], v[:, 3:6], v[:, 6:9]
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(acc).all(1) | ~np.isfinite(gyr).all(1), RowFlag.NON_FINITE)
    _flag(
        flags,
        (np.linalg.norm(acc, axis=1) > lim.max_specific_force_mps2)
        | (np.abs(gyr) > lim.max_angular_rate_radps).any(1),
        RowFlag.OUT_OF_RANGE,
    )
    return ImuRecord(t_us=t, flags=flags, mag=mag.copy(), acc=acc.copy(), gyr=gyr.copy())


def load_fog(path: Path) -> FogRecord:
    """Load ``kvh.csv``: utime, integrated heading angle (rad)."""
    t, v = _read(path, 2)
    angle = v[:, 0].copy()
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(angle), RowFlag.NON_FINITE)
    return FogRecord(t_us=t, flags=flags, angle=angle)


def load_wheels(path: Path, limits: RawLimits | None = None) -> WheelRecord:
    """Load ``wheels.csv``: utime, left speed, right speed (m/s)."""
    lim = limits or RawLimits()
    t, v = _read(path, 3)
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(v).all(1), RowFlag.NON_FINITE)
    _flag(flags, (np.abs(v) > lim.max_wheel_speed_mps).any(1), RowFlag.OUT_OF_RANGE)
    return WheelRecord(t_us=t, flags=flags, left=v[:, 0].copy(), right=v[:, 1].copy())


def load_gps(path: Path, limits: RawLimits | None = None) -> GpsRecord:
    """Load ``gps.csv`` or ``gps_rtk.csv``: utime, mode, num_sats, lat, lon, alt, track, speed."""
    lim = limits or RawLimits()
    t, v = _read(path, 8)
    mode = np.nan_to_num(v[:, 0], nan=-1).astype(np.int64)
    lat, lon, alt = v[:, 2].copy(), v[:, 3].copy(), v[:, 4].copy()
    flags = _time_flags(t)
    _flag(flags, ~(np.isfinite(lat) & np.isfinite(lon)), RowFlag.NON_FINITE)
    _flag(
        flags,
        (np.abs(lat) > np.pi / 2) | (np.abs(lon) > np.pi),
        RowFlag.OUT_OF_RANGE,
    )
    _flag(flags, mode < lim.min_gps_fix_mode, RowFlag.GPS_NO_FIX)
    held = np.zeros(len(t), dtype=bool)
    if len(t) > 1:
        held[1:] = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1]) & (alt[1:] == alt[:-1])
    _flag(flags, held, RowFlag.GPS_HELD)
    return GpsRecord(
        t_us=t,
        flags=flags,
        mode=mode,
        num_sats=np.nan_to_num(v[:, 1], nan=-1).astype(np.int64),
        lat=lat,
        lon=lon,
        alt=alt,
        track=v[:, 5].copy(),
        speed=v[:, 6].copy(),
    )


def load_euler(path: Path) -> EulerRecord:
    """Load ``ms25_euler.csv``: utime, roll, pitch, heading (rad). Diagnostic only."""
    t, v = _read(path, 4)
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(v).all(1), RowFlag.NON_FINITE)
    return EulerRecord(t_us=t, flags=flags, rpy=v.copy())


def load_odometry(path: Path) -> OdometryRecord:
    """Load ``odometry_mu.csv`` / ``odometry_mu_100hz.csv``: utime, x, y, z, roll, pitch, yaw."""
    t, v = _read(path, 7)
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(v).all(1), RowFlag.NON_FINITE)
    return OdometryRecord(t_us=t, flags=flags, pose=v.copy())


def load_rtk_error(path: Path) -> RtkErrorRecord:
    """Load ``gps_rtk_err.csv``: utime, error (m). Diagnostic only."""
    t, v = _read(path, 2)
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(v[:, 0]), RowFlag.NON_FINITE)
    return RtkErrorRecord(t_us=t, flags=flags, err=v[:, 0].copy())
