"""Compare error statistics of real sessions with their simulated twins (Phase 2, step 11).

    uv run python scripts/validate_sim_realism.py [scenario]

Writes reports/phase2_eval/sim_realism_<scenario>.csv (per session, real vs sim) and prints
the session-median ratio sim/real per statistic.
"""

from __future__ import annotations

import sys

import pandas as pd

from lase.data.calibration import load_calibration
from lase.data.session import load_inputs, load_reference_for
from lase.data.splits import Split, load_split
from lase.sim.config import load_scenario
from lase.sim.fit import session_stats
from lase.sim.simulate import simulate_session
from lase.utils.paths import repo_root


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "realistic"
    cfg = load_scenario(scenario)
    cal = load_calibration()
    rows = []
    dates = load_split().dates(Split.TRAIN)[::4]  # every 4th training day (6 sessions)
    for i, d in enumerate(dates):
        real = session_stats(
            load_inputs(d), load_reference_for(d), cal, cal.wheels.speed_scale or 1.0
        )
        run = simulate_session(d, cfg.model_copy(update={"seed": i}))
        sim = session_stats(run.streams, run.reference, cal, cfg.wheels.speed_scale)
        for k in real:
            rows.append({"date": d.isoformat(), "stat": k, "real": real[k], "sim": sim[k]})
        print(d, flush=True)
    df = pd.DataFrame(rows)
    out = repo_root() / "reports" / "phase2_eval" / f"sim_realism_{scenario}.csv"
    df.to_csv(out, index=False, float_format="%.5g")
    summary = df.groupby("stat")[["real", "sim"]].median()
    summary["sim/real"] = summary["sim"] / summary["real"]
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
