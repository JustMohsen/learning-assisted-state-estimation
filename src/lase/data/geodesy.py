"""GPS coordinates <-> NCLT local NED frame.

The dataset's local frame is *defined* by the linearisation in the paper (Sec. 3,
Eqs. 1-3, Table 3); ground truth is expressed in it. :class:`LocalProjection`
implements exactly that, so converted GPS fixes and ground truth share a frame by
construction. :func:`geodetic_to_ned_wgs84` is an exact ECEF-based transform kept as
an independent cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lase.data.calibration import LocalFrame

F64 = npt.NDArray[np.float64]

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


@dataclass(frozen=True, slots=True)
class LocalProjection:
    """NCLT linearised projection around a fixed origin (angles in radians internally)."""

    lat0: float
    lon0: float
    alt0: float
    r_ns: float  # north-south (meridian) radius at the origin
    r_ew: float  # east-west (prime vertical) radius at the origin

    @classmethod
    def from_config(cls, frame: LocalFrame) -> LocalProjection:
        """Build from the calibration config, computing radii with paper Eq. 1."""
        lat0 = np.deg2rad(frame.lat_origin_deg)
        re, rp = frame.earth_equatorial_radius_m, frame.earth_polar_radius_m
        denom = (re * np.cos(lat0)) ** 2 + (rp * np.sin(lat0)) ** 2
        r_ns = (re * rp) ** 2 / denom**1.5
        r_ew = re**2 / np.sqrt(denom)
        return cls(
            lat0=float(lat0),
            lon0=float(np.deg2rad(frame.lon_origin_deg)),
            alt0=frame.alt_origin_m,
            r_ns=float(r_ns),
            r_ew=float(r_ew),
        )

    def to_local(self, lat: F64, lon: F64, alt: F64) -> F64:
        """Latitude/longitude (rad) and altitude (m) to local ``(N, 3)`` [north, east, down] (Eq. 2)."""
        x = np.sin(np.asarray(lat) - self.lat0) * self.r_ns
        y = np.sin(np.asarray(lon) - self.lon0) * self.r_ew * np.cos(self.lat0)
        z = self.alt0 - np.asarray(alt)
        return np.stack([x, y, z], axis=-1)

    def to_geodetic(self, ned: F64) -> tuple[F64, F64, F64]:
        """Inverse of :meth:`to_local` (Eq. 3): returns lat (rad), lon (rad), alt (m)."""
        lat = np.arcsin(ned[..., 0] / self.r_ns) + self.lat0
        lon = np.arcsin(ned[..., 1] / (self.r_ew * np.cos(self.lat0))) + self.lon0
        alt = self.alt0 - ned[..., 2]
        return lat, lon, alt


def _geodetic_to_ecef(lat: F64, lon: F64, alt: F64) -> F64:
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - e2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def geodetic_to_ned_wgs84(
    lat: F64, lon: F64, alt: F64, lat0: float, lon0: float, alt0: float
) -> F64:
    """Exact WGS-84 geodetic (rad, m) to NED at the origin ``(lat0, lon0, alt0)``. Cross-check only."""
    lat, lon, alt = np.asarray(lat), np.asarray(lon), np.asarray(alt)
    d = _geodetic_to_ecef(lat, lon, alt) - _geodetic_to_ecef(
        np.asarray(lat0), np.asarray(lon0), np.asarray(alt0)
    )
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    R = np.array([[-sl * co, -sl * so, cl], [-so, co, 0.0], [-cl * co, -cl * so, -sl]])
    ned: F64 = d @ R.T
    return ned
