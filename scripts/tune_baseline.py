"""Grid search for the analytical baseline on TRAINING days only (Phase 3, step 9).

    uv run python scripts/tune_baseline.py

Each configuration runs on 30-minute segments of six training sessions with GPS withheld
in the injected outage windows, so one run scores both GPS-aided accuracy and outage drift.
Writes reports/phase3_baseline/tuning_runs.csv and tuning_summary.csv.
"""

from __future__ import annotations

import itertools
import sys
import time

import pandas as pd

from lase.data.splits import Split, load_split
from lase.filter.experiment import run_many
from lase.filter.params import FilterConfig, load_filter_config
from lase.utils.paths import repo_root


def configs(stage: str = "1") -> dict[str, FilterConfig]:
    """Stage 1: coarse grid. Stage 2: extend past the stage-1 edges, plus the wheel-scale state.

    All stages start from configs/parameters.yaml and change only the gridded values.
    """
    base = load_filter_config("baseline")
    out = {}
    if stage == "r7a":  # revision: GPS handling x FOG bias model (on configs/parameters.yaml)
        gps = {
            "gpsW5_1hz": {"error_state": False},
            "gpsGM_7hz": {"error_state": True, "min_interval_s": 0.0},
            "gpsGM_1hz": {"error_state": True, "min_interval_s": 1.0},
        }
        fog = {
            "fogOld": (1e-5, 1e-7),
            "fog1e-4_rw1e-6": (1e-4, 1e-6),
            "fog1e-4_rw2e-6": (1e-4, 2e-6),
        }
        for (gn, gu), (fn, (fp, frw)) in itertools.product(gps.items(), fog.items()):
            name = f"r7a_{gn}_{fn}"
            out[name] = base.model_copy(
                update={
                    "name": name,
                    "gnss": base.gnss.model_copy(update=gu),
                    "prior": base.prior.model_copy(update={"fog_bias_radps": fp}),
                    "noise": base.noise.model_copy(update={"fog_bias_rw": frw}),
                }
            )
        # control: Earth-rate compensation off (expected no better once the FOG prior is realistic)
        c = out["r7a_gpsGM_1hz_fog1e-4_rw1e-6"]
        out["r7a_gpsGM_1hz_fog1e-4_rw1e-6_earthOff"] = c.model_copy(
            update={"name": c.name + "_earthOff", "earth_rotation": False}
        )
        return out
    if stage == "r7b":  # revision: lateral-constraint sigma x encoder term, on the r7a winner
        win = load_filter_config(_arg(2, "baseline"))
        for nhc, enc in itertools.product((0.05, 0.1, 0.2), (0.0, 0.24)):
            name = f"r7b_nhc{nhc:g}_enc{enc:g}"
            out[name] = win.model_copy(
                update={
                    "name": name,
                    "nonholonomic": win.nonholonomic.model_copy(update={"sigma_lateral_mps": nhc}),
                    "wheel": win.wheel.model_copy(update={"encoder_pitch_coef_m": enc}),
                }
            )
        return out
    if stage == "3":  # GPS recovery rule variants on the stage-2 winner (set below)
        best = _cfg(base, "s3", 5.0, 1.0, float(_arg(2, "0.15")), float(_arg(3, "0.05")))
        for after in (90.0, 300.0, 1e9):
            name = f"s3_recovery{after:g}"
            out[name] = best.model_copy(
                update={
                    "name": name,
                    "gnss": best.gnss.model_copy(update={"recovery_after_s": after}),
                }
            )
        return out
    if stage == "2":
        for an, ws in itertools.product((0.05, 0.1, 0.2), (0.15, 0.3)):
            name = f"s2_gps5_int1_wh{ws:g}_acc{an:g}"
            out[name] = _cfg(base, name, 5.0, 1.0, ws, an)
        name = "s2_gps5_int1_wh0.15_acc0.05_wscale"
        c = _cfg(base, name, 5.0, 1.0, 0.15, 0.05)
        out[name] = c.model_copy(
            update={
                "prior": c.prior.model_copy(update={"wheel_scale": 0.02}),
                "noise": c.noise.model_copy(update={"wheel_scale_rw": 1e-5}),
            }
        )
        return out
    for gs, gi, ws, an in itertools.product((3.0, 5.0), (0.0, 1.0), (0.05, 0.15), (0.02, 0.05)):
        name = f"gps{gs:g}_int{gi:g}_wh{ws:g}_acc{an:g}"
        out[name] = _cfg(base, name, gs, gi, ws, an)
    return out


def _arg(i: int, default: str) -> str:
    return sys.argv[i] if len(sys.argv) > i else default


def _cfg(base: FilterConfig, name: str, gs: float, gi: float, ws: float, an: float) -> FilterConfig:
    return base.model_copy(
        update={
            "name": name,
            "nonholonomic": base.nonholonomic.model_copy(update={"enabled": True}),
            "gnss": base.gnss.model_copy(update={"sigma_m": gs, "min_interval_s": gi}),
            "wheel": base.wheel.model_copy(update={"sigma_mps": ws}),
            "noise": base.noise.model_copy(update={"acc_noise_density": an}),
        }
    )


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    dates = load_split().dates(Split.TRAIN)[::4][:6]  # 6 training sessions across the year
    cfgs = configs(stage)
    jobs, tags = [], []
    for name, cfg in cfgs.items():
        for d in dates:
            jobs.append((d, cfg, "outages", 30.0, 5.0))  # skip the first 5 min (start-up)
            tags.append(name)
    t0 = time.time()
    rows = run_many(jobs, workers=8)
    df = pd.DataFrame(rows)
    df.insert(0, "config", tags)
    out = repo_root() / "reports" / "phase3_baseline"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if stage == "1" else f"_stage{stage}"
    # r7 stages: also save which session subset was used
    df.to_csv(out / f"tuning_runs{suffix}.csv", index=False, float_format="%.5g")
    keys = [
        "nodes_pos_median_m",
        "nodes_pos_p95_m",
        "nodes_heading_rmse_deg",
        "dense_vel_rmse_mps",
        "inj_outage_final_med_m",
        "wheel_nis_mean",
        "gnss_nis_mean",
    ]
    summ = df.groupby("config")[[k for k in keys if k in df]].median()
    # rank: normalised sum of position p95, heading RMSE and outage end error (lower is better)
    score = 0.0
    for k in ("nodes_pos_p95_m", "nodes_heading_rmse_deg", "inj_outage_final_med_m"):
        if k in summ:
            score = score + summ[k] / summ[k].median()
    summ["score"] = score
    summ = summ.sort_values("score")
    summ.to_csv(out / f"tuning_summary{suffix}.csv", float_format="%.5g")
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
