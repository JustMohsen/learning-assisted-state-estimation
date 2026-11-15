"""Run a filter configuration on every simulation scenario (Phase 3).

    uv run python scripts/sim_baseline.py [filter_config_name]

Writes reports/phase3_baseline/sim_scenarios_<name>.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from lase.filter.params import load_filter_config
from lase.filter.validation import run_sim_scenario
from lase.utils.paths import repo_root

SCENARIOS = ["clean", "biased", "matched", "nominal", "slip", "gps_degraded", "realistic"]


def _one(args: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    scenario, cfg_name = args
    r = run_sim_scenario(scenario, cfg=load_filter_config(cfg_name))
    return scenario, {
        "metrics": r["metrics"],
        "nees_pos": asdict(r["nees_pos"]),
        "nis_wheel": asdict(r["nis_wheel"]),
        "nis_gnss": asdict(r["nis_gnss"]),
        "wheel_accept": r["wheel_accept"],
        "gnss_accept": r["gnss_accept"],
        "gyr_bias_rmse_2nd_half": r["gyr_bias_rmse_2nd_half"].tolist(),
        "gyr_bias_true_rms": r["gyr_bias_true_rms"].tolist(),
        "acc_bias_rmse_2nd_half": r["acc_bias_rmse_2nd_half"].tolist(),
        "acc_bias_true_rms": r["acc_bias_true_rms"].tolist(),
        "us_per_step": r["us_per_step"],
    }


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    with ProcessPoolExecutor(7) as ex:
        out = dict(ex.map(_one, [(s, name) for s in SCENARIOS]))
    path = repo_root() / "reports" / "phase3_baseline" / f"sim_scenarios_{name}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        m = v["metrics"]
        print(
            f"{k:13s} pos RMSE {m['pos_rmse_m']:7.3f} med {m['pos_median_m']:6.2f} p95 {m['pos_p95_m']:6.2f} "
            f"| head {m['heading_rmse_deg']:5.2f} | vel {m['vel_rmse_mps']:.3f} | NEES {v['nees_pos']['mean']:7.1f} "
            f"| NIS w {v['nis_wheel']['mean']:.2f} g {v['nis_gnss']['mean']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
