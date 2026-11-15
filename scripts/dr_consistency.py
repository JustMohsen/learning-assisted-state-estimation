"""Dead-reckoning consistency: are the filter's uncertainties honest without GPS? (R8)

    uv run python scripts/dr_consistency.py [variant ...]

Runs the filter with GPS disabled from several reference-initialised starts per training
session and compares the errors with the claimed covariance as time goes on:
- position NEES (2 dof, ideal 2) and heading NEES (1 dof, ideal 1);
- along-track and cross-track normalised errors (ideal RMS 1): along-track shows speed/scale
  errors, cross-track shows heading errors.
A value growing above its ideal with time means dead reckoning is overconfident.

Variants (configs/parameters.yaml plus the change named): see ``variants()``.
Writes reports/phase3_baseline/dr_consistency_runs.csv and _summary.csv.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from lase.data.reference import Reference
from lase.data.session import load_reference_for
from lase.data.splits import Split, load_split
from lase.evaluation.estimate import NavEstimate
from lase.filter.experiment import run_many
from lase.filter.params import FilterConfig, load_filter_config
from lase.utils.paths import repo_root

HORIZONS_S = (10, 30, 60, 120, 300, 600)
STARTS_MIN = (5.0, 15.0, 25.0, 35.0)
RUN_MIN = 10.0


def variants() -> dict[str, FilterConfig]:
    base = load_filter_config("baseline")
    out = {"baseline": base}
    out["scale_sigma0"] = base.model_copy(
        update={"prior": base.prior.model_copy(update={"wheel_scale": 0.0})}
    )
    for rw in (2e-6, 5e-6):
        out[f"fog_rw{rw:g}"] = base.model_copy(
            update={"noise": base.noise.model_copy(update={"fog_bias_rw": rw})}
        )
    for ws in (0.08, 0.3):
        out[f"wheel_sigma{ws:g}"] = base.model_copy(
            update={"wheel": base.wheel.model_copy(update={"sigma_mps": ws})}
        )
    for an in (0.2,):
        out[f"acc_noise{an:g}"] = base.model_copy(
            update={"noise": base.noise.model_copy(update={"acc_noise_density": an})}
        )
    return out


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def normalised_errors(est: NavEstimate, ref: Reference) -> dict[str, np.ndarray]:
    """Per-output-sample position/heading NEES and along/cross-track normalised errors."""
    gt = ref.gt
    ok = gt.valid
    t_ref = gt.t_us[ok]
    p_ref = np.c_[
        np.interp(est.t_us, t_ref, gt.p_w[ok, 0]), np.interp(est.t_us, t_ref, gt.p_w[ok, 1])
    ]
    yaw_ref = np.interp(est.t_us, t_ref, np.unwrap(gt.rpy[ok, 2]))
    assert est.cov is not None
    e = est.pos_ne - p_ref
    Pp = est.cov[:, :2, :2]
    nees_pos = np.einsum("ni,nij,nj->n", e, np.linalg.inv(Pp), e)
    eh = _wrap(est.heading - yaw_ref)
    nees_head = eh**2 / est.cov[:, 4, 4]
    u = np.c_[np.cos(yaw_ref), np.sin(yaw_ref)]  # travel direction (north, east)
    n = np.c_[-u[:, 1], u[:, 0]]
    out = {"t_us": est.t_us, "nees_pos": nees_pos, "nees_head": nees_head}
    for name, d in (("along", u), ("cross", n)):
        var = np.einsum("ni,nij,nj->n", d, Pp, d)
        out[f"z_{name}"] = (e * d).sum(1) / np.sqrt(var)
        out[f"err_{name}"] = (e * d).sum(1)
    return out


def main() -> int:
    names = sys.argv[1:] or list(variants())
    cfgs = {k: v for k, v in variants().items() if k in names}
    dates = load_split().dates(Split.TRAIN)[::4][:6]  # same sessions as the tuning grids
    jobs: list[tuple[Any, ...]] = []
    tags = []
    for name, cfg in cfgs.items():
        for d in dates:
            for s in STARTS_MIN:
                jobs.append((d, cfg, "nogps", RUN_MIN, s, True))
                tags.append((name, d, s))
    t0 = time.time()
    results = run_many(jobs, workers=8)
    refs = {d: load_reference_for(d) for d in dates}
    rows = []
    for (name, d, s), res in zip(tags, results, strict=True):
        if res is None or not isinstance(res, tuple):
            continue
        est = res[2]
        if len(est) == 0:
            continue
        ne = normalised_errors(est, refs[d])
        t = (ne["t_us"] - ne["t_us"][0]) * 1e-6
        for h in HORIZONS_S:
            i = int(np.searchsorted(t, h))
            if i >= len(t) or abs(t[i] - h) > 1.0:
                continue
            rows.append(
                {
                    "variant": name,
                    "date": d.isoformat(),
                    "start_min": s,
                    "horizon_s": h,
                    **{k: float(v[i]) for k, v in ne.items() if k != "t_us"},
                }
            )
    df = pd.DataFrame(rows)
    out = repo_root() / "reports" / "phase3_baseline"
    df.to_csv(out / "dr_consistency_runs.csv", index=False, float_format="%.5g")
    g = df.groupby(["variant", "horizon_s"])
    summary = pd.DataFrame(
        {
            "n": g.size(),
            "nees_pos_mean": g["nees_pos"].mean(),  # ideal 2
            "nees_pos_median": g["nees_pos"].median(),  # ideal ~1.39 (chi2, 2 dof)
            "nees_head_mean": g["nees_head"].mean(),  # ideal 1
            "z_along_rms": g["z_along"].apply(lambda x: float(np.sqrt((x**2).mean()))),
            "z_cross_rms": g["z_cross"].apply(lambda x: float(np.sqrt((x**2).mean()))),
            "err_along_rms_m": g["err_along"].apply(lambda x: float(np.sqrt((x**2).mean()))),
            "err_cross_rms_m": g["err_cross"].apply(lambda x: float(np.sqrt((x**2).mean()))),
        }
    ).reset_index()
    summary.to_csv(out / "dr_consistency_summary.csv", index=False, float_format="%.4g")
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    print(summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
