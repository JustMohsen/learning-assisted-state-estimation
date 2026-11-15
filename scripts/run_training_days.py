"""Run the tuned baseline on ALL training days, full sessions, with and without GPS.

    uv run python scripts/run_training_days.py [filter_config_name]

Diagnostic for the Phase 3 review: how often do GPS re-acquisition failures and FOG
heading drift occur? Writes reports/phase3_baseline/training_full_runs.csv.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from lase.data.splits import Split, load_split
from lase.filter.experiment import run_many
from lase.filter.params import load_filter_config
from lase.utils.paths import repo_root


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    cfg = load_filter_config(name)
    jobs = [
        (d, cfg, mode, None, 0.0)
        for mode in ("gps", "nogps")
        for d in load_split().dates(Split.TRAIN)
    ]
    t0 = time.time()
    df = pd.DataFrame(run_many(jobs, workers=8))
    out = repo_root() / "reports" / "phase3_baseline" / "training_full_runs.csv"
    df.to_csv(out, index=False, float_format="%.5g")
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    cols = [
        "date",
        "mode",
        "nodes_pos_median_m",
        "nodes_pos_p95_m",
        "nodes_pos_max_m",
        "nodes_heading_rmse_deg",
        "gnss_accept",
        "gnss_recoveries",
    ]
    print(df[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
