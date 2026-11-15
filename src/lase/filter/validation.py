"""Filter validation helpers (Phase 3): Monte Carlo covariance check and simulation runs."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from lase.data.calibration import load_calibration
from lase.evaluation.consistency import nis, position_nees
from lase.evaluation.estimate import reference_samples
from lase.evaluation.metrics import compute_errors, summarize
from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.propagation import ImuInterval, boxminus, propagate
from lase.filter.runner import initial_condition_from_reference, run_filter
from lase.filter.so3 import matrix_to_quat, so3_exp
from lase.filter.state import N_ERR, NavState
from lase.sim.config import load_scenario
from lase.sim.simulate import simulate_session

D = torch.float64


@dataclass(frozen=True)
class MonteCarloCheck:
    """Ratio of Monte Carlo error variance to predicted variance per error coordinate."""

    n_runs: int
    duration_s: float
    var_ratio: np.ndarray  # (9,) for position, velocity, attitude


def monte_carlo_propagation(
    n_runs: int = 400, duration_s: float = 10.0, dt: float = 0.02, seed: int = 0
) -> MonteCarloCheck:
    """Propagate noisy IMU/FOG data (white noise per the filter densities) without updates.

    Biases are held fixed (zero random walk, zero prior) so only white-noise propagation
    is tested. Returns MC variance / predicted variance for p, v, theta (ideal: 1).
    """
    cfg = FilterConfig()
    noise = cfg.noise.model_copy(
        update={"acc_bias_rw": 0.0, "gyr_bias_rw": 0.0, "fog_bias_rw": 0.0}
    )
    params = FilterParams.build(load_calibration(), cfg)
    g = torch.Generator().manual_seed(seed)
    B = n_runs
    C0 = so3_exp(torch.tensor([[0.05, -0.03, 1.0]], dtype=D))
    x0 = NavState(
        p_w=torch.zeros(B, 3, dtype=D),
        v_w=torch.tensor([[1.0, 0.5, 0.0]], dtype=D).repeat(B, 1),
        q_wb=matrix_to_quat(C0).repeat(B, 1),
        b_acc=torch.zeros(B, 3, dtype=D),
        b_gyr=torch.zeros(B, 3, dtype=D),
        b_fog=torch.zeros(B, 1, dtype=D),
        wheel_scale=torch.ones(B, 1, dtype=D),
        gps_err=torch.zeros(B, 2, dtype=D),
    )
    f_true = torch.tensor([[0.5, 0.2, -9.8]], dtype=D)
    w_true = torch.tensor([[0.1, -0.05, 0.2]], dtype=D)
    sa = noise.acc_noise_density / np.sqrt(dt)
    sg = noise.gyr_noise_density / np.sqrt(dt)
    sf = noise.fog_angle_random_walk * np.sqrt(dt)
    learned = HeadOutputs.neutral(B)
    clean, noisy = x0, x0
    P = torch.zeros(B, N_ERR, N_ERR, dtype=D)
    for _ in range(int(duration_s / dt)):
        na = sa * torch.randn(B, 3, generator=g, dtype=D)
        ng = sg * torch.randn(B, 3, generator=g, dtype=D)
        nf = sf * torch.randn(B, generator=g, dtype=D)
        z_rate = w_true[0, 2]
        base = {"dt": torch.full((B,), dt, dtype=D), "fog_valid": torch.ones(B, dtype=D)}
        iv_clean = ImuInterval(
            acc0=f_true.repeat(B, 1),
            acc1=f_true.repeat(B, 1),
            gyr0=w_true.repeat(B, 1),
            gyr1=w_true.repeat(B, 1),
            fog_dangle=torch.full((B,), -float(z_rate) * dt, dtype=D),
            **base,
        )
        iv_noisy = ImuInterval(
            acc0=f_true + na,
            acc1=f_true + na,
            gyr0=w_true + ng,
            gyr1=w_true + ng,
            fog_dangle=-(z_rate * dt + nf),
            **base,
        )
        clean, _, _ = propagate(clean, P, iv_clean, params, noise, learned)
        noisy, P, _ = propagate(noisy, P, iv_noisy, params, noise, learned)
    err = boxminus(noisy, clean)[:, :9].numpy()
    pred = P[0].diagonal()[:9].numpy()
    return MonteCarloCheck(n_runs, duration_s, err.var(0) / pred)


def run_sim_scenario(
    scenario: str, date: dt.date = dt.date(2013, 1, 10), cfg: FilterConfig | None = None
) -> dict[str, Any]:
    """Run the baseline on a simulated session and return accuracy/consistency/bias results."""
    cfg = cfg or FilterConfig()
    run = simulate_session(date, load_scenario(scenario))
    params = replace(
        FilterParams.build(load_calibration(), cfg), wheel_scale=run.truth.wheel_speed_scale
    )
    ic = initial_condition_from_reference(run.reference)
    res = run_filter(run.streams, ic, cfg, params)
    ref = reference_samples(run.reference, "dense", t_window=(ic.t_us, int(res.estimate.t_us[-1])))
    e = compute_errors(res.estimate, ref)
    imu_t = run.streams.imu.t_us
    j = np.searchsorted(imu_t, res.biases["t_us"]).clip(0, len(imu_t) - 1)
    half = len(j) // 2
    gb_err = res.biases["gyr"][half:] - run.truth.imu_gyr_bias[j[half:]]
    ab_err = res.biases["acc"][half:] - run.truth.imu_acc_bias[j[half:]]
    return {
        "scenario": scenario,
        "metrics": summarize(e),
        "nees_pos": position_nees(res.estimate, e),
        "nis_wheel": nis(res.wheel["innovation"], res.wheel["S"]),
        "nis_gnss": nis(res.gnss["innovation"], res.gnss["S"]),
        "wheel_accept": float(res.wheel["accepted"].mean()),
        "gnss_accept": float(res.gnss["accepted"].mean()),
        "gyr_bias_rmse_2nd_half": np.sqrt((gb_err**2).mean(0)),
        "gyr_bias_true_rms": np.sqrt((run.truth.imu_gyr_bias**2).mean(0)),
        "acc_bias_rmse_2nd_half": np.sqrt((ab_err**2).mean(0)),
        "acc_bias_true_rms": np.sqrt((run.truth.imu_acc_bias**2).mean(0)),
        "wall_s": res.wall_time_s,
        "us_per_step": 1e6 * res.wall_time_s / res.n_imu_steps,
    }
