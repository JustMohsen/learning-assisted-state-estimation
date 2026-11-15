"""Simulate a session: truth trajectory from a real reference, sensors at real timestamps.

Outputs the same types as the real-data loaders (:class:`SensorStreams`,
:class:`Reference`), plus :class:`SimTruth` with every hidden quantity, so all
downstream code runs unchanged on simulated data.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from lase.data.calibration import NcltCalibration, load_calibration
from lase.data.conversions import unwrap_fog
from lase.data.geodesy import LocalProjection
from lase.data.raw import F64, I64, B, make_fog, make_gps, make_imu, make_wheels
from lase.data.reference import GraphNodes, GroundTruth, Reference
from lase.data.session import SensorStreams, load_inputs, load_reference_for
from lase.geometry.constants import earth_rate_ned
from lase.geometry.euler import wrap_angle
from lase.sim.config import SimConfig
from lase.sim.sensors import Span, simulate_fog, simulate_gps, simulate_imu, simulate_wheels
from lase.sim.trajectory import TruthTrajectory, truth_from_reference


@dataclass(frozen=True)
class SimTruth:
    """Hidden truth of a simulation run (never an estimator input)."""

    config: SimConfig
    trajectory: TruthTrajectory
    imu_acc_bias: F64  # (N_imu, 3) at IMU sample times
    imu_gyr_bias: F64
    imu_acc_scale: F64  # (3,)
    imu_gyr_scale: F64
    fog_bias: F64  # (N_fog,) rad/s
    wheel_speed_scale: float
    track_width_m: float
    wheel_slip: B  # (N_wheel,)
    slip_spans: list[Span]
    gps_error_ned: F64  # (N_gps, 3)
    gps_sigma_mult: F64
    gps_outlier: B
    gps_held: B
    gps_outage_spans: list[Span]
    gps_episode_spans: list[Span]


@dataclass(frozen=True)
class SimRun:
    """Simulated inputs, reference built from the truth, and the truth itself."""

    streams: SensorStreams
    reference: Reference
    truth: SimTruth


def _reference_from_truth(
    traj: TruthTrajectory, rate_hz: float = 100.0, node_dt_s: float = 3.6
) -> Reference:
    step = round(1e6 / rate_hz)
    t = np.arange(traj.t0_us, traj.t1_us + 1, step, dtype=np.int64)
    st = traj.at(t)
    rpy = st.rpy.copy()
    rpy[:, 2] = wrap_angle(rpy[:, 2])
    gt = GroundTruth(t_us=t, p_w=st.p_w, rpy=rpy, flags=np.zeros(len(t), dtype=np.uint16))
    k = max(1, round(node_dt_s * rate_hz))
    idx = np.arange(0, len(t), k, dtype=np.int64)
    cov = np.tile(np.eye(6) * 1e-8, (len(idx), 1, 1))
    nodes = GraphNodes(
        t_us=t[idx],
        cov=cov,
        gt_index=idx,
        gt_offset_us=np.zeros(len(idx), dtype=np.int64),
        valid=np.ones(len(idx), dtype=bool),
    )
    return Reference(gt=gt, nodes=nodes)


def simulate_session(
    date: dt.date,
    config: SimConfig,
    t0_us: int | None = None,
    t1_us: int | None = None,
    calibration: NcltCalibration | None = None,
) -> SimRun:
    """Simulate one session (optionally a window) under ``config``.

    Sensor timestamps are the real session's timestamps inside the window, so the
    simulated timing, jitter and (optionally) natural GPS gaps are realistic.
    """
    cal = calibration or load_calibration()
    rng = np.random.default_rng(config.seed)
    real = load_inputs(date, cal)
    traj = truth_from_reference(load_reference_for(date), config.trajectory, t0_us, t1_us)

    def inside(t: I64) -> I64:
        out: I64 = t[traj.contains(t)]
        return out

    t_imu, t_fog, t_wh = inside(real.imu.t_us), inside(real.fog.t_us), inside(real.wheels.t_us)
    if config.use_real_gps_gaps:
        t_gps = inside(real.gps.t_us[real.gps.valid])
    else:
        t_gps = np.arange(traj.t0_us, traj.t1_us, round(1e6 / 7.0), dtype=np.int64)

    w_ie = (
        np.asarray(earth_rate_ned(cal.local_frame.lat_origin_deg))
        if config.earth_rotation
        else None
    )
    imu = simulate_imu(traj, t_imu, cal, config.imu, rng, w_ie)
    fog = simulate_fog(traj, t_fog, config.fog, rng, w_ie)
    wh = simulate_wheels(traj, t_wh, config.wheels, rng)
    gps = simulate_gps(traj, t_gps, cal, config.gps, rng)

    n_g = len(gps.t_us)
    nan = np.full(n_g, np.nan)
    gps_rec = make_gps(
        gps.t_us, gps.mode, np.zeros(n_g, dtype=np.int64), gps.lat, gps.lon, gps.alt, nan, nan
    )
    proj = LocalProjection.from_config(cal.local_frame)
    streams = SensorStreams(
        date=date,
        imu=make_imu(t_imu, np.zeros((len(t_imu), 3)), imu.acc, imu.gyr),
        fog=make_fog(t_fog, fog.angle_released),
        wheels=make_wheels(t_wh, wh.left, wh.right),
        gps=gps_rec,
        fog_angle=unwrap_fog(fog.angle_released),
        gnss_ned=proj.to_local(gps.lat, gps.lon, gps.alt),
    )
    truth = SimTruth(
        config=config,
        trajectory=traj,
        imu_acc_bias=imu.acc_bias,
        imu_gyr_bias=imu.gyr_bias,
        imu_acc_scale=imu.acc_scale,
        imu_gyr_scale=imu.gyr_scale,
        fog_bias=fog.bias,
        wheel_speed_scale=config.wheels.speed_scale,
        track_width_m=config.wheels.track_width_m,
        wheel_slip=wh.slip,
        slip_spans=wh.slip_spans,
        gps_error_ned=gps.error_ned,
        gps_sigma_mult=gps.sigma_mult,
        gps_outlier=gps.outlier,
        gps_held=gps.held,
        gps_outage_spans=gps.outage_spans,
        gps_episode_spans=gps.episode_spans,
    )
    return SimRun(streams=streams, reference=_reference_from_truth(traj), truth=truth)
