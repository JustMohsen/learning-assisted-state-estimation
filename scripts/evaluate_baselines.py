"""Score the trivial baselines on all sessions with the Phase 2 evaluation harness.

    uv run python scripts/evaluate_baselines.py

Writes reports/phase2_eval/: per-session metrics, pooled metrics, outage curves and figures.
These numbers are reference points for the Phase 3 analytical baseline.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from lase.data.calibration import load_calibration
from lase.data.session import load_diagnostics, load_inputs, load_reference_for
from lase.data.splits import load_split
from lase.evaluation.baselines import (
    gps_baseline,
    gps_hold_baseline,
    odometry_baseline,
    odometry_reanchored,
)
from lase.evaluation.estimate import reference_samples
from lase.evaluation.metrics import compute_errors, relative_pose_error, summarize
from lase.evaluation.outages import OutageCurves, outage_curves, plan_outages, withhold_gps
from lase.evaluation.plots import plot_errors_vs_time, plot_outage_growth, plot_trajectories
from lase.utils.paths import repo_root


def main() -> int:
    out = repo_root() / "reports" / "phase2_eval"
    out.mkdir(parents=True, exist_ok=True)
    cal = load_calibration()
    split = load_split()
    rows = []
    pooled: dict[str, list] = {}
    curves: dict[str, list[OutageCurves]] = {"GPS hold": [], "odometry (oracle-anchored)": []}
    all_spans = {}
    for d in split.dates():
        s, ref, diag = load_inputs(d), load_reference_for(d), load_diagnostics(d)
        nodes, dense = reference_samples(ref, "nodes"), reference_samples(ref, "dense")
        spans = plan_outages(s, seed=int(d.strftime("%Y%m%d")))
        all_spans[d.isoformat()] = spans
        ests = [gps_baseline(s, cal), odometry_baseline(diag.odometry_100hz, ref)]
        for est in ests:
            e_nodes = compute_errors(est, nodes)
            pooled.setdefault(est.name, []).append(e_nodes)
            row = {"date": d.isoformat(), "split": split.split_of(d).value, "method": est.name}
            row.update({f"nodes_{k}": v for k, v in summarize(e_nodes).items()})
            row.update({f"dense_{k}": v for k, v in summarize(compute_errors(est, dense)).items()})
            for delta in (10.0, 60.0):
                r = relative_pose_error(est, dense, delta)
                row[f"rpe{int(delta)}_trans_rmse_m"] = r.trans_rmse_m
                row[f"rpe{int(delta)}_heading_rmse_deg"] = r.heading_rmse_deg
            rows.append(row)
        # outage curves: GPS withheld in the planned windows (hold baseline repeats last fix)
        hold = gps_hold_baseline(withhold_gps(s, spans), cal)
        reanch = odometry_reanchored(diag.odometry_100hz, ref, spans)
        for est in (hold, reanch):
            curves[est.name].append(outage_curves(compute_errors(est, dense), spans))
        if d == split.dates()[2]:
            plot_trajectories(nodes, ests, out / f"trajectories_{d}.png", f"Baselines {d}")
            plot_errors_vs_time(
                {e.name: compute_errors(e, dense) for e in ests},
                out / f"errors_{d}.png",
                title=f"Horizontal error, {d}",
            )
        print(f"{d}: {len(spans)} outage windows", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "baselines_by_session.csv", index=False, float_format="%.4g")
    pooled_rows = []
    for name, series in pooled.items():
        for sp in ("train", "validation", "test"):
            idx = [i for i, r in enumerate(r for r in rows if r["method"] == name) if r]
            sel = [
                e
                for e, r in zip(series, [r for r in rows if r["method"] == name], strict=True)
                if r["split"] == sp
            ]
            pooled_rows.append({"method": name, "split": sp, **summarize(sel)})
            _ = idx
    pd.DataFrame(pooled_rows).to_csv(out / "baselines_pooled.csv", index=False, float_format="%.4g")
    merged = {}
    for name, cs in curves.items():
        cs = [c for c in cs if len(c.errors_m)]
        merged[name] = OutageCurves(
            offsets_s=cs[0].offsets_s,
            errors_m=np.concatenate([c.errors_m for c in cs]),
            durations_s=np.concatenate([c.durations_s for c in cs]),
            final_error_m=np.concatenate([c.final_error_m for c in cs]),
        )
    plot_outage_growth(merged, out / "outage_growth.png", "Injected GPS outages, all sessions")
    summary = {
        name: {
            "n_outages": len(c.errors_m),
            "offsets_s": c.offsets_s.tolist(),
            "median_m": np.round(c.percentile(50), 3).tolist(),
            "p90_m": np.round(c.percentile(90), 3).tolist(),
        }
        for name, c in merged.items()
    }
    (out / "outage_curves.json").write_text(json.dumps(summary, indent=1))
    (out / "outage_spans.json").write_text(json.dumps(all_spans, indent=1))
    print(pd.DataFrame(pooled_rows).round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
