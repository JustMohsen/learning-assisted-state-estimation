"""Evaluate the tuned analytical baseline on the VALIDATION days (Phase 3, step 10).

    uv run python scripts/evaluate_baseline.py [filter_config_name]

Runs the baseline with and without the soft non-holonomic constraint in three modes
(all sensors; injected GPS outages; no GPS) on both validation days, alongside the
Phase 2 reference points, and writes reports/phase3_baseline/validation_*.
Test days are not touched.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from lase.data.calibration import load_calibration
from lase.data.session import load_diagnostics, load_inputs, load_reference_for
from lase.data.splits import Split, load_split
from lase.evaluation.baselines import gps_baseline, odometry_baseline, odometry_reanchored
from lase.evaluation.outages import OutageCurves
from lase.evaluation.plots import plot_errors_vs_time, plot_outage_growth, plot_trajectories
from lase.evaluation.report import evaluate_estimate
from lase.filter.experiment import run_many
from lase.filter.params import load_filter_config
from lase.utils.paths import repo_root


def _merge(curves: list[OutageCurves]) -> OutageCurves | None:
    cs = [c for c in curves if c is not None and len(c.errors_m)]
    if not cs:
        return None
    return OutageCurves(
        offsets_s=cs[0].offsets_s,
        errors_m=np.concatenate([c.errors_m for c in cs]),
        durations_s=np.concatenate([c.durations_s for c in cs]),
        final_error_m=np.concatenate([c.final_error_m for c in cs]),
    )


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    tuned = load_filter_config(name)
    no_nhc = tuned.model_copy(
        update={
            "name": tuned.name + "-no-nhc",
            "nonholonomic": tuned.nonholonomic.model_copy(update={"enabled": False}),
        }
    )
    dates = load_split().dates(Split.VALIDATION)
    jobs, tags = [], []
    for cfg in (tuned, no_nhc):
        for mode in ("gps", "outages", "nogps"):
            for d in dates:
                jobs.append((d, cfg, mode, None, 0.0, True))
                tags.append(cfg.name)
    results = run_many(jobs, workers=8)
    out = repo_root() / "reports" / "phase3_baseline"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, (row, _rep, _est, _sp, _nat) in zip(tags, results, strict=True):
        rows.append({"config": tag, **row})
    # Phase 2 reference points on the same days and windows
    cal = load_calibration()
    ref_curves: dict[str, list[OutageCurves]] = {"odometry (oracle-anchored)": []}
    for d in dates:
        s, ref, diag = load_inputs(d), load_reference_for(d), load_diagnostics(d)
        t0 = int(ref.gt.t_us[ref.gt.valid][0]) + 11_000_000
        win = (t0, int(s.imu.t_us[-1]))
        for est in (gps_baseline(s, cal), odometry_baseline(diag.odometry_100hz, ref)):
            rows.append(
                {
                    "config": est.name,
                    "date": d.isoformat(),
                    "mode": "reference point",
                    **evaluate_estimate(est, ref, t_window=win).row(),
                }
            )
        spans = next(
            r[3]
            for (_t, r) in zip(tags, results, strict=True)
            if r[0]["date"] == d.isoformat() and r[0]["mode"] == "outages"
        )
        if spans:
            reanch = odometry_reanchored(diag.odometry_100hz, ref, spans)
            rep = evaluate_estimate(reanch, ref, t_window=win, injected_spans=spans)
            ref_curves["odometry (oracle-anchored)"].append(rep.injected_outages)  # type: ignore[arg-type]
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_runs.csv", index=False, float_format="%.5g")

    # outage curves: tuned vs no-NHC vs oracle-anchored odometry
    curves = {}
    for cfg_name in (tuned.name, no_nhc.name):
        cs = [
            r[1].injected_outages
            for t, r in zip(tags, results, strict=True)
            if t == cfg_name and r[0]["mode"] == "outages"
        ]
        merged = _merge(cs)
        if merged is not None:
            curves[cfg_name] = merged
    oracle = _merge(ref_curves["odometry (oracle-anchored)"])
    if oracle is not None:
        curves["odometry (oracle-anchored, diagnostic)"] = oracle
    plot_outage_growth(
        curves, out / "validation_outage_growth.png", "Injected GPS outages, validation days"
    )
    json.dump(
        {
            k: {
                "n": len(c.errors_m),
                "offsets_s": c.offsets_s.tolist(),
                "median_m": np.round(c.percentile(50), 3).tolist(),
                "p90_m": np.round(c.percentile(90), 3).tolist(),
            }
            for k, c in curves.items()
        },
        (out / "validation_outage_curves.json").open("w"),
        indent=1,
    )
    # figures for the longer validation day
    d = dates[-1]
    s, ref = load_inputs(d), load_reference_for(d)
    pick = {
        r[0]["mode"]: r
        for t, r in zip(tags, results, strict=True)
        if t == tuned.name and r[0]["date"] == d.isoformat()
    }
    from lase.evaluation.estimate import reference_samples

    nodes = reference_samples(ref, "nodes")
    plot_trajectories(
        nodes,
        [pick["gps"][2], pick["nogps"][2]],
        out / f"validation_trajectories_{d}.png",
        f"Baseline {tuned.name}, {d}",
    )
    plot_errors_vs_time(
        {
            "all sensors": pick["gps"][1].dense_errors,
            "injected outages": pick["outages"][1].dense_errors,
            "no GPS": pick["nogps"][1].dense_errors,
        },
        out / f"validation_errors_{d}.png",
        spans=pick["outages"][3],
        title=f"Horizontal error, {d} (shaded: injected outages)",
    )
    cols = [
        "config",
        "date",
        "mode",
        "nodes_pos_rmse_m",
        "nodes_pos_median_m",
        "nodes_pos_p95_m",
        "nodes_pos_max_m",
        "nodes_heading_rmse_deg",
        "dense_vel_rmse_mps",
        "rpe10s_nodes_trans_rmse_m",
        "rpe60s_nodes_trans_rmse_m",
        "inj_outage_med_at_60s_m",
        "inj_outage_med_at_120s_m",
        "nat_outage_final_med_m",
        "gnss_nis_mean",
        "wheel_nis_mean",
        "nees_pos_mean",
        "us_per_step",
    ]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
