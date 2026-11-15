"""Per-session data access, split by role.

Three separate entry points enforce the sensor contract:

- :func:`load_inputs`: estimator inputs only (IMU, FOG, wheels, consumer GPS);
- :func:`load_reference_for`: reference trajectory (training/evaluation only);
- :func:`load_diagnostics`: vendor attitude, RTK and supplied odometry (analysis only).

Estimator code receives :class:`SensorStreams` and nothing else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from lase.data.calibration import NcltCalibration, load_calibration
from lase.data.conversions import unwrap_fog
from lase.data.geodesy import LocalProjection
from lase.data.raw import (
    F64,
    EulerRecord,
    FogRecord,
    GpsRecord,
    ImuRecord,
    OdometryRecord,
    RawLimits,
    RtkErrorRecord,
    WheelRecord,
    load_euler,
    load_fog,
    load_gps,
    load_imu,
    load_odometry,
    load_rtk_error,
    load_wheels,
)
from lase.data.reference import Reference, load_reference
from lase.utils.paths import nclt_dir


def extracted_dir(date: dt.date) -> Path:
    """Directory of the extracted sensor CSVs for ``date``."""
    return nclt_dir() / "extracted" / date.isoformat()


@dataclass(frozen=True, slots=True)
class SensorStreams:
    """Estimator inputs for one session, with causal derived quantities.

    ``fog_angle`` is the unwrapped FOG angle (unwrapping uses only past samples).
    ``gnss_ned`` is each GPS fix (antenna position) in the dataset's local NED frame.
    """

    date: dt.date
    imu: ImuRecord
    fog: FogRecord
    wheels: WheelRecord
    gps: GpsRecord
    fog_angle: F64  # (N_fog,)
    gnss_ned: F64  # (N_gps, 3)


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Diagnostic streams (never estimator inputs)."""

    euler: EulerRecord
    rtk: GpsRecord
    rtk_err: RtkErrorRecord
    odometry_100hz: OdometryRecord


def load_inputs(
    date: dt.date,
    calibration: NcltCalibration | None = None,
    limits: RawLimits | None = None,
) -> SensorStreams:
    """Load the estimator-input streams for ``date``."""
    cal = calibration or load_calibration()
    d = extracted_dir(date)
    fog = load_fog(d / "kvh.csv")
    gps = load_gps(d / "gps.csv", limits)
    proj = LocalProjection.from_config(cal.local_frame)
    return SensorStreams(
        date=date,
        imu=load_imu(d / "ms25.csv", limits),
        fog=fog,
        wheels=load_wheels(d / "wheels.csv", limits),
        gps=gps,
        fog_angle=unwrap_fog(fog.angle),
        gnss_ned=proj.to_local(gps.lat, gps.lon, gps.alt),
    )


def load_reference_for(date: dt.date) -> Reference:
    """Load the reference trajectory for ``date`` (training/evaluation only)."""
    root = nclt_dir() / "raw"
    return load_reference(
        root / "ground_truth" / f"groundtruth_{date.isoformat()}.csv",
        root / "covariance" / f"cov_{date.isoformat()}.csv",
    )


def load_diagnostics(date: dt.date) -> Diagnostics:
    """Load diagnostic streams for ``date`` (analysis only)."""
    d = extracted_dir(date)
    return Diagnostics(
        euler=load_euler(d / "ms25_euler.csv"),
        rtk=load_gps(d / "gps_rtk.csv"),
        rtk_err=load_rtk_error(d / "gps_rtk_err.csv"),
        odometry_100hz=load_odometry(d / "odometry_mu_100hz.csv"),
    )
