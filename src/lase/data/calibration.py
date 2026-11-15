"""NCLT frame and sensor-mounting calibration (validated config with sources)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from lase.geometry.euler import F64, euler_zyx_to_matrix
from lase.utils.config import Config, load_config
from lase.utils.paths import repo_root

DEFAULT_CALIBRATION_PATH = Path("configs/data/nclt_calibration.yaml")
Status = Literal["published", "estimated", "to_estimate", "to_verify"]


class LocalFrame(Config):
    """Linearised local NED frame of the dataset (paper Sec. 3, Table 3)."""

    description: str
    lat_origin_deg: float
    lon_origin_deg: float
    alt_origin_m: float
    earth_equatorial_radius_m: float
    earth_polar_radius_m: float
    source: str


class BodyFrame(Config):
    """Body frame definition (documentation only)."""

    description: str
    source: str


class SensorMount(Config):
    """Sensor pose in the body frame: position ``xyz_m`` and orientation ``rpy_deg``."""

    xyz_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float] | None
    status: Status
    source: str

    @property
    def l_b(self) -> F64:
        """Lever arm from the body origin to the sensor, in body axes (m)."""
        return np.asarray(self.xyz_m, dtype=np.float64)

    @property
    def C_bs(self) -> F64:
        """Rotation mapping sensor-frame vectors into the body frame (identity if not applicable)."""
        if self.rpy_deg is None:
            return np.eye(3)
        r, p, y = np.deg2rad(self.rpy_deg)
        return euler_zyx_to_matrix(r, p, y)


class Sensors(Config):
    """Mounting of the sensors used by the project."""

    imu: SensorMount
    fog: SensorMount
    gps: SensorMount
    rtk: SensorMount


class Wheels(Config):
    """Wheel geometry."""

    speed_reference: str
    track_width_m: float | None
    track_width_status: Status
    source: str


class Timing(Config):
    """Timing notes per sensor (documentation)."""

    fog: str
    others: str


class NcltCalibration(Config):
    """Complete NCLT calibration config."""

    local_frame: LocalFrame
    body_frame: BodyFrame
    sensors: Sensors
    wheels: Wheels
    timing: Timing


def load_calibration(path: str | Path | None = None) -> NcltCalibration:
    """Load the calibration config (default: ``configs/data/nclt_calibration.yaml``)."""
    return load_config(path or repo_root() / DEFAULT_CALIBRATION_PATH, NcltCalibration)
