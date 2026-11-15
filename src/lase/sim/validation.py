"""Simulator validation checks S1-S3 (plans/PHASE_2_SIM_EVAL.md, section 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lase.data.calibration import NcltCalibration
from lase.data.raw import F64, I64
from lase.sim.config import ImuErrors
from lase.sim.sensors import simulate_imu
from lase.sim.strapdown import strapdown
from lase.sim.trajectory import TruthTrajectory

_ZERO_IMU = ImuErrors(
    acc_noise_mps2=0,
    gyr_noise_radps=0,
    acc_bias_init_mps2=0,
    gyr_bias_init_radps=0,
    acc_bias_rw_mps2_sqrts=0,
    gyr_bias_rw_radps_sqrts=0,
)


@dataclass(frozen=True)
class StrapdownCheck:
    """S1: open-loop integration of clean IMU data over windows."""

    window_s: float
    native_rate_hz: float
    native_err_m: float  # median final position error over windows
    fast_err_m: float  # same at 4x the native rate
    convergence_ratio: float  # native / fast (about 16 for a second-order scheme)


def _window_errors(
    traj: TruthTrajectory, cal: NcltCalibration, starts: I64, window_s: float, rate_hz: float
) -> F64:
    errs = []
    lever = cal.sensors.imu.l_b
    for t0 in starts:
        t = np.arange(t0, t0 + int(window_s * 1e6) + 1, round(1e6 / rate_hz), dtype=np.int64)
        imu = simulate_imu(traj, t, cal, _ZERO_IMU, np.random.default_rng(0))
        st = traj.at(t)
        omega_l = np.cross(st.omega_b, lever)
        p_imu = st.p_w + st.C_wb @ lever
        v_imu = st.v_w + np.einsum("qij,qj->qi", st.C_wb, omega_l)
        p, _, _ = strapdown(t, imu.acc, imu.gyr, p_imu[0], v_imu[0], st.C_wb[0])
        errs.append(float(np.linalg.norm(p[-1] - p_imu[-1])))
    return np.asarray(errs)


def check_strapdown(
    traj: TruthTrajectory,
    cal: NcltCalibration,
    native_rate_hz: float = 47.0,
    window_s: float = 10.0,
    n_windows: int = 20,
) -> StrapdownCheck:
    """Integrate clean simulated IMU data from the true initial state (S1)."""
    span = traj.t1_us - traj.t0_us - int((window_s + 20) * 1e6)
    starts = traj.t0_us + int(10e6) + np.linspace(0, span, n_windows).astype(np.int64)
    e1 = _window_errors(traj, cal, starts, window_s, native_rate_hz)
    e4 = _window_errors(traj, cal, starts, window_s, 4 * native_rate_hz)
    m1, m4 = float(np.median(e1)), float(np.median(e4))
    return StrapdownCheck(window_s, native_rate_hz, m1, m4, m1 / max(m4, 1e-12))


@dataclass(frozen=True)
class NoiseCheck:
    """S3: realised noise statistics vs configuration."""

    acc_noise_std: float
    acc_noise_cfg: float
    gyr_noise_std: float
    gyr_noise_cfg: float


def check_imu_noise(
    traj: TruthTrajectory, cal: NcltCalibration, t_us: I64, err: ImuErrors, seed: int = 0
) -> NoiseCheck:
    """Realised IMU white-noise standard deviation after removing truth and injected bias."""
    clean = simulate_imu(traj, t_us, cal, _ZERO_IMU, np.random.default_rng(seed))
    noisy = simulate_imu(traj, t_us, cal, err, np.random.default_rng(seed))
    na = noisy.acc - (1 + noisy.acc_scale) * clean.acc - noisy.acc_bias
    ng = noisy.gyr - (1 + noisy.gyr_scale) * clean.gyr - noisy.gyr_bias
    return NoiseCheck(float(na.std()), err.acc_noise_mps2, float(ng.std()), err.gyr_noise_radps)
