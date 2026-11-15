"""Isolated 2x2 test of the R1/R2 wheel-model changes on training days.

    uv run python scripts/wheel_model_ab.py

Factors: encoder pitch-rate term (0 vs 0.24 m) x vertical non-holonomic constraint (on vs off).
All variants use the pitch-independent rolling speed and otherwise configs/parameters.yaml.
Writes reports/phase3_baseline/wheel_model_ab.csv.
"""

from __future__ import annotations

import itertools
import sys
import time

import pandas as pd

from lase.data.splits import Split, load_split
from lase.filter.experiment import run_many
from lase.filter.params import load_filter_config
from lase.utils.paths import repo_root


def main() -> int:
    base = load_filter_config("baseline")
    dates = load_split().dates(Split.TRAIN)[::4][:6]  # same sessions as the tuning grids
    jobs, tags = [], []
    for enc, vert in itertools.product((0.0, 0.24), (True, False)):
        name = f"enc{enc:g}_vertical{'On' if vert else 'Off'}"
        cfg = base.model_copy(
            update={
                "name": name,
                "wheel": base.wheel.model_copy(update={"encoder_pitch_coef_m": enc}),
                "nonholonomic": base.nonholonomic.model_copy(update={"vertical_enabled": vert}),
            }
        )
        for d in dates:
            jobs.append((d, cfg, "outages", 30.0, 5.0))
            tags.append(name)
    t0 = time.time()
    df = pd.DataFrame(run_many(jobs, workers=6))
    df.insert(0, "variant", tags)
    df.to_csv(
        repo_root() / "reports" / "phase3_baseline" / "wheel_model_ab.csv",
        index=False,
        float_format="%.5g",
    )
    keys = [
        "nodes_pos_median_m",
        "nodes_pos_p95_m",
        "nodes_heading_rmse_deg",
        "dense_vel_rmse_mps",
        "rpe60s_trans_rmse_m",
        "inj_outage_final_med_m",
        "wheel_nis_mean",
    ]
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    print(df.groupby("variant")[keys].median().round(3).to_string())
    print(
        df.pivot_table(index="date", columns="variant", values="inj_outage_final_med_m")
        .round(2)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
