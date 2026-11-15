"""Fit simulator error models to real training-day data and write configs/sim/realistic.yaml.

uv run python scripts/fit_sim_errors.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from lase.data.calibration import load_calibration
from lase.data.session import load_inputs, load_reference_for
from lase.data.splits import Split, load_split
from lase.sim.fit import (
    GpsStats,
    WheelStats,
    accumulate_gps,
    accumulate_wheels,
    at_rest_imu_std,
    fit_gauss_markov,
)
from lase.utils.paths import repo_root


def main() -> int:
    cal = load_calibration()
    scale = cal.wheels.speed_scale or 1.0
    gs, ws = GpsStats(), WheelStats()
    rest_a, rest_g = [], []
    for d in load_split().dates(Split.TRAIN):
        s, ref = load_inputs(d), load_reference_for(d)
        accumulate_gps(gs, s, ref, cal)
        accumulate_wheels(ws, s, ref, scale)
        a, g, n = at_rest_imu_std(s)
        if n:
            rest_a.append(a)
            rest_g.append(g)
        print(d, flush=True)
    sw, sgm, tau = fit_gauss_markov(gs.acf_num, gs.acf_den)
    mod = np.array(gs.moderate)
    ep = np.array(gs.episodes) if gs.episodes else np.zeros((0, 2))
    res = np.concatenate(ws.resid)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    ev = np.array(ws.events) if ws.events else np.zeros((0, 2))
    out = {
        "gps": {
            "hours": gs.hours,
            "fixes": gs.n_fixes,
            "inlier_sigma_per_axis_m": float(np.sqrt(np.median(gs.inlier_var_ne))),
            "white_sigma_m": sw,
            "gm_sigma_m": sgm,
            "gm_tau_s": tau,
            "acf_1hz_lags_0_10_60_120_300": [
                float(gs.acf_num[k] / max(gs.acf_den[k], 1)) for k in (0, 10, 60, 120, 300)
            ],
            "sigma_mode2_m": float(np.sqrt(np.median(gs.var_by_mode[2])))
            if gs.var_by_mode[2]
            else None,
            "sigma_mode3_m": float(np.sqrt(np.median(gs.var_by_mode[3])))
            if gs.var_by_mode[3]
            else None,
            "moderate_outlier_prob_10_50m": len(mod) / gs.n_fixes,
            "moderate_outlier_rms_m": float(np.sqrt((mod**2).mean())) if len(mod) else 0.0,
            "gross_episodes": len(ep),
            "gross_episode_rate_per_hour": len(ep) / gs.hours,
            "gross_episode_duration_s_median_p90": [
                float(np.median(ep[:, 0])),
                float(np.percentile(ep[:, 0], 90)),
            ]
            if len(ep)
            else None,
            "gross_episode_offset_m_median_p90_max": [
                float(np.median(ep[:, 1])),
                float(np.percentile(ep[:, 1], 90)),
                float(ep[:, 1].max()),
            ]
            if len(ep)
            else None,
            "mode2_frac": gs.mode2_fixes / gs.n_fixes,
            "mode2_dwell_s_mean": float(np.mean(gs.mode2_runs_s)),
            "mode3_dwell_s_mean": float(np.mean(gs.mode3_runs_s)),
            "held_frac": gs.held / gs.n_fixes,
        },
        "wheels": {
            "moving_minutes": ws.moving_min,
            "residual_std_mps": float(res.std()),
            "residual_robust_std_mps": float(mad),
            "residual_mean_mps": float(res.mean()),
            "excursion_rate_per_moving_min": len(ev) / ws.moving_min,
            "excursion_duration_s_median_p90": [
                float(np.median(ev[:, 0])),
                float(np.percentile(ev[:, 0], 90)),
            ]
            if len(ev)
            else None,
            "excursion_relative_magnitude_median_p90": [
                float(np.median(ev[:, 1])),
                float(np.percentile(ev[:, 1], 90)),
            ]
            if len(ev)
            else None,
        },
        "imu_at_rest": {
            "n_sessions": len(rest_a),
            "acc_std_mps2": np.median(np.array(rest_a), 0).tolist(),
            "gyr_std_radps": np.median(np.array(rest_g), 0).tolist(),
        },
    }
    path = repo_root() / "reports" / "phase2_eval" / "sim_fit.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
