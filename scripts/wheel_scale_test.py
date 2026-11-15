"""R6: wheel scale carried across sessions (chained slow state) vs fixed 1.054, training days.

    uv run python scripts/wheel_scale_test.py [minutes_per_session]

Chains run sessions in date order (sequential); variants run in parallel processes.
Writes reports/phase3_baseline/wheel_scale_test.csv.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from lase.data.splits import Split, load_split
from lase.filter.experiment import _chain_worker, run_many
from lase.filter.params import load_filter_config
from lase.parameters import load_parameters
from lase.utils.paths import repo_root


def main() -> int:
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    dates = list(load_split().dates(Split.TRAIN))
    base = load_filter_config("baseline")
    chain_on = base.wheel_scale_chain.model_copy(update={"enabled": True})
    variants = {
        "chain_rw1e-6": base.model_copy(
            update={
                "wheel_scale_chain": chain_on,
                "noise": base.noise.model_copy(update={"wheel_scale_rw": 1e-6}),
            }
        ),
        "chain_rw1e-5": base.model_copy(
            update={
                "wheel_scale_chain": chain_on,
                "noise": base.noise.model_copy(update={"wheel_scale_rw": 1e-5}),
            }
        ),
    }
    t0 = time.time()
    out = repo_root() / "reports" / "phase3_baseline" / "wheel_scale_test.csv"
    partial = out.with_suffix(".partial.csv")
    partial.unlink(missing_ok=True)
    params = load_parameters()  # snapshot: the YAML is not re-read while jobs run
    rows = []
    with ProcessPoolExecutor(len(variants)) as ex:
        futs = {
            name: ex.submit(
                _chain_worker, (dates, cfg, "outages", minutes, params, str(partial), name)
            )
            for name, cfg in variants.items()
        }
        fixed = run_many([(d, base, "outages", minutes) for d in dates], workers=4, params=params)
        for r in fixed:
            rows.append({"variant": "fixed_1.054", **r})
        pd.DataFrame(rows).to_csv(partial, mode="a", header=not partial.exists(), index=False)
        for f in futs.values():
            rows.extend(f.result())
    df = pd.DataFrame(rows)
    df.to_csv(
        out, index=False, float_format="%.5g"
    )  # compare to the reference-measured per-session scale (Phase 1, V3)
    v3 = pd.read_csv(repo_root() / "reports" / "phase1_audit" / "validation.csv")[
        ["date", "v3_scale"]
    ]
    df = df.merge(v3, on="date", how="left")
    df["scale_err"] = df["final_wheel_scale"] - df["v3_scale"]
    print(f"{len(df)} runs in {time.time() - t0:.0f} s")
    keys = [
        "nodes_pos_median_m",
        "nodes_pos_p95_m",
        "nodes_heading_rmse_deg",
        "inj_outage_final_med_m",
        "wheel_nis_mean",
    ]
    print(df.groupby("variant")[keys].median().round(3).to_string())
    print(df.groupby("variant")["scale_err"].describe().round(4).to_string())
    print(
        df[df.variant != "fixed_1.054"][
            ["variant", "date", "init_wheel_scale", "final_wheel_scale", "v3_scale"]
        ]
        .round(4)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
