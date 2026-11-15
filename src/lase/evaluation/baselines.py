"""Trivial reference-point estimators for the evaluation harness.

- :func:`gps_baseline`: raw consumer-GPS track (causal): antenna position minus the
  lever arm rotated by course, held between fixes; velocity and course from fixes
  at least ~1 s apart. Course is not body heading when reversing or slipping.
- :func:`odometry_baseline`: NCLT's supplied odometry (fused wheels/FOG/IMU, no GPS),
  aligned to the reference by the common initial-pose prior (agreed in the scope).
"""

from __future__ import annotations

import numpy as np

from lase.data.calibration import NcltCalibration
from lase.data.raw import F64, OdometryRecord
from lase.data.reference import Reference
from lase.data.session import SensorStreams
from lase.evaluation.estimate import NavEstimate
from lase.geometry.euler import wrap_angle


def gps_baseline(
    streams: SensorStreams, cal: NcltCalibration, min_dt_s: float = 1.0, min_speed_mps: float = 0.3
) -> NavEstimate:
    """Raw GPS track as a causal estimate."""
    g = streams.gps
    m = g.valid
    t = g.t_us[m]
    p = streams.gnss_ned[m][:, :2]
    n = len(t)
    vel = np.full((n, 2), np.nan)
    j = 0
    for i in range(n):  # causal backward difference over >= min_dt_s
        while j < i and (t[i] - t[j + 1]) >= min_dt_s * 1e6:
            j += 1
        if (t[i] - t[j]) >= min_dt_s * 1e6:
            vel[i] = (p[i] - p[j]) / ((t[i] - t[j]) * 1e-6)
    heading = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        if np.isfinite(vel[i]).all() and np.linalg.norm(vel[i]) > min_speed_mps:
            last = float(np.arctan2(vel[i, 1], vel[i, 0]))
        heading[i] = last
    lx, ly = cal.sensors.gps.l_b[:2]
    c, s = np.cos(np.nan_to_num(heading)), np.sin(np.nan_to_num(heading))
    lever_ne = np.stack([c * lx - s * ly, s * lx + c * ly], axis=1)
    pos = p - np.where(np.isfinite(heading)[:, None], lever_ne, 0.0)
    return NavEstimate("raw GPS", t, pos, vel, wrap_angle(heading))


def odometry_baseline(odo: OdometryRecord, ref: Reference, vel_dt_s: float = 0.5) -> NavEstimate:
    """NCLT odometry aligned (SE(2)) to the reference pose at the first common time."""
    m = odo.valid
    t, pose = odo.t_us[m], odo.pose[m]
    tg = ref.gt.t_us[ref.gt.valid]
    k0 = int(np.searchsorted(t, tg[0]))
    t, pose = t[k0:], pose[k0:]
    p_ref0, rpy_ref0 = ref.pose_at(t[:1])
    dpsi = rpy_ref0[0, 2] - pose[0, 5]
    c, s = np.cos(dpsi), np.sin(dpsi)
    R = np.array([[c, -s], [s, c]])
    xy = (pose[:, :2] - pose[0, :2]) @ R.T + p_ref0[0, :2]
    heading = wrap_angle(pose[:, 5] + dpsi)
    vel = np.full((len(t), 2), np.nan)
    lag = np.searchsorted(t, t - int(vel_dt_s * 1e6), side="left")
    ok = (t - t[lag]) > 0
    vel[ok] = (xy[ok] - xy[lag[ok]]) / ((t[ok] - t[lag[ok]]) * 1e-6)[:, None]
    est: F64 = xy
    return NavEstimate("NCLT odometry", t, est, vel, heading)


def gps_hold_baseline(
    streams: SensorStreams, cal: NcltCalibration, rate_hz: float = 10.0
) -> NavEstimate:
    """Raw GPS held at a fixed output rate (causal): during outages the last fix is repeated.

    Its outage error is roughly the distance travelled: a 'no dead reckoning' bound.
    """
    base = gps_baseline(streams, cal)
    t = np.arange(base.t_us[0], streams.imu.t_us[-1], round(1e6 / rate_hz), dtype=np.int64)
    k = np.searchsorted(base.t_us, t, side="right") - 1
    return NavEstimate("GPS hold", t, base.pos_ne[k], base.vel_ne[k], base.heading[k])


def odometry_reanchored(
    odo: OdometryRecord, ref: Reference, spans: list[tuple[int, int]]
) -> NavEstimate:
    """*Diagnostic, non-causal:* NCLT odometry restarted from the true pose at each span start.

    Shows the dead-reckoning error growth that wheels + FOG achieve from a perfect start;
    a reference point for outage curves, not a comparable method.
    """
    m = odo.valid
    t_all, pose_all = odo.t_us[m], odo.pose[m]
    ts, ps, hs = [], [], []
    for a, b in spans:
        k = (t_all >= a) & (t_all <= b)
        t, pose = t_all[k], pose_all[k]
        if len(t) < 2:
            continue
        p_ref0, rpy_ref0 = ref.pose_at(t[:1])
        dpsi = rpy_ref0[0, 2] - pose[0, 5]
        c, s = np.cos(dpsi), np.sin(dpsi)
        R = np.array([[c, -s], [s, c]])
        ts.append(t)
        ps.append((pose[:, :2] - pose[0, :2]) @ R.T + p_ref0[0, :2])
        hs.append(wrap_angle(pose[:, 5] + dpsi))
    t = np.concatenate(ts) if ts else np.zeros(0, dtype=np.int64)
    n = len(t)
    return NavEstimate(
        "odometry (oracle-anchored)",
        t,
        np.concatenate(ps) if ps else np.zeros((0, 2)),
        np.full((n, 2), np.nan),
        np.concatenate(hs) if hs else np.zeros(0),
    )
