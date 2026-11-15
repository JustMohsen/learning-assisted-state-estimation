"""Phase 4 pilot: train method B (GPS, wheel, lateral sigma) and evaluate it on validation days.

    uv run python scripts/phase4_pilot.py [all|cache|train|eval|compare]

1. cache: run baseline A over the pilot training days, saving state/covariance snapshots
   every ``training.pilot.snapshot_every_s`` (chunk starts).
2. train: AdamW on the three deltas; each step averages chunk gradients from every
   session (one process per session); anchor penalty / number of chunks.
3. eval: B = baseline A with the three learned sigmas, run through
   ``lase.filter.experiment.run_session`` on the validation days (gps and outages modes).

Outputs in runs/phase4_pilot/ (cache, log) and reports/phase4_pilot/ (results). Test days
are not touched. ``configs/parameters.yaml`` is read once at start.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch

from lase.data.splits import Split, load_split
from lase.filter.experiment import run_session
from lase.filter.params import FilterConfig
from lase.parameters import Parameters, load_parameters
from lase.training.pilot import LEARNED, anchor_sigmas, anchors, cache_snapshots, chunk_grad
from lase.utils.paths import repo_root

RUNS = repo_root() / "runs" / "phase4_pilot"
OUT = repo_root() / "reports" / "phase4_pilot"


def pilot_dates(p: Parameters) -> list[Any]:
    train = sorted(load_split().dates(Split.TRAIN))
    n = min(p.training.pilot.n_sessions, len(train))
    idx = np.unique(np.linspace(0, len(train) - 1, n).round().astype(int))
    return [train[i] for i in idx]


def cache(p: Parameters) -> None:
    dates = pilot_dates(p)
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(dates), p.training.pilot.workers)) as ex:
        for d, snaps in zip(dates, ex.map(cache_snapshots, dates, [p] * len(dates)), strict=True):
            with (RUNS / f"snapshots_{d.isoformat()}.pkl").open("wb") as f:
                pickle.dump(snaps, f)
            print(f"cache {d}: {len(snaps)} snapshots ({time.time() - t0:.0f}s)", flush=True)


def lr_at(step: int, p: Parameters) -> float:
    pl = p.training.pilot
    lo = p.training.optimizer.lr_final
    if step < pl.warmup_steps:
        return pl.lr * (step + 1) / pl.warmup_steps
    frac = (step - pl.warmup_steps) / max(pl.steps - pl.warmup_steps, 1)
    return lo + 0.5 * (pl.lr - lo) * (1 + math.cos(math.pi * frac))


def train(p: Parameters) -> np.ndarray:
    pl, w = p.training.pilot, p.training.windows
    chunk_s = w.burn_in_s + w.loss_window_s
    snaps: dict[Any, list[Any]] = {}
    for d in pilot_dates(p):
        with (RUNS / f"snapshots_{d.isoformat()}.pkl").open("rb") as f:
            all_s = pickle.load(f)
        # a chunk needs its full length before the last snapshot (end of the session)
        t_last = all_s[-1].t_us
        snaps[d] = [s for s in all_s if s.t_us + chunk_s * 1e6 < t_last]
    n_chunks = sum(len(v) for v in snaps.values())
    per = max(1, pl.chunks_per_step // len(snaps))
    rng = np.random.default_rng(0)
    delta = torch.zeros(len(LEARNED), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.AdamW([delta], lr=pl.lr, weight_decay=p.training.optimizer.weight_decay)
    sig = anchor_sigmas(p)
    wpen = p.training.loss.anchor_penalty_weight
    pools = {d: ProcessPoolExecutor(max_workers=1) for d in snaps}
    log = []
    print(
        f"train: {len(snaps)} sessions, {n_chunks} chunks, {per} per session per step", flush=True
    )
    try:
        for step in range(pl.steps):
            t0 = time.time()
            futs = []
            for d, ss in snaps.items():
                for i in rng.choice(len(ss), size=per, replace=False):
                    args = (d, ss[i], delta.detach().tolist(), p)
                    futs.append(pools[d].submit(chunk_grad, args))
            res = [f.result() for f in futs]
            ok = [r for r in res if r["n"] > 0]
            task = float(np.mean([r["loss"] for r in ok]))
            g_task = np.mean([r["grad"] for r in ok], axis=0)
            opt.zero_grad()
            pen = wpen * ((delta / sig) ** 2).sum() / n_chunks
            pen.backward()  # type: ignore[no-untyped-call]
            assert delta.grad is not None
            delta.grad += torch.as_tensor(g_task)
            gnorm = float(
                torch.nn.utils.clip_grad_norm_([delta], p.training.optimizer.grad_clip_norm)
            )
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, p)
            opt.step()
            row = {
                "step": step,
                "task_loss": task,
                "penalty": float(pen.detach()),
                "grad_norm": gnorm,
                "lr": lr_at(step, p),
                "wall_s": time.time() - t0,
                **{f"delta_{k}": float(v) for k, v in zip(LEARNED, delta.detach(), strict=True)},
                **{f"grad_{k}": float(v) for k, v in zip(LEARNED, g_task, strict=True)},
            }
            log.append(row)
            pd.DataFrame(log).to_csv(RUNS / "train_log.csv", index=False)
            print(
                f"step {step:3d} loss {task:.4f} |g| {gnorm:.3f} delta "
                f"{np.round(delta.detach().numpy(), 4)} ({row['wall_s']:.0f}s)",
                flush=True,
            )
    finally:
        for ex in pools.values():
            ex.shutdown()
    out: np.ndarray = delta.detach().numpy()
    return out


def b_config(cfg: FilterConfig, delta: np.ndarray) -> FilterConfig:
    a = anchors(cfg)
    s = {k: a[k] * float(np.exp(v)) for k, v in zip(LEARNED, delta, strict=True)}
    return cfg.model_copy(
        update={
            "name": "pilot-b",
            "gnss": cfg.gnss.model_copy(update={"sigma_m": s["gnss_sigma"]}),
            "wheel": cfg.wheel.model_copy(update={"sigma_mps": s["wheel_sigma"]}),
            "nonholonomic": cfg.nonholonomic.model_copy(
                update={"sigma_lateral_mps": s["lateral_sigma"]}
            ),
        }
    )


def _eval_job(args: tuple[Any, ...]) -> Any:
    torch.set_num_threads(1)
    d, cfg, mode, p = args
    return run_session(d, cfg, mode, params=p)


def evaluate(p: Parameters, delta: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    b = b_config(p.filter, delta)
    learned = {
        k: {
            "anchor": anchors(p.filter)[k],
            "delta": float(v),
            "value": anchors(p.filter)[k] * float(np.exp(v)),
        }
        for k, v in zip(LEARNED, delta, strict=True)
    }
    json.dump(learned, (OUT / "learned.json").open("w"), indent=1)
    print(json.dumps(learned, indent=1), flush=True)
    dates = load_split().dates(Split.VALIDATION)
    jobs = [(d, b, m, p) for m in ("outages", "gps") for d in dates]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        rows = list(ex.map(_eval_job, jobs))
    pd.DataFrame(rows).assign(config="pilot-b").to_csv(OUT / "validation_runs_b.csv", index=False)
    print(f"wrote {OUT / 'validation_runs_b.csv'}", flush=True)


METRICS = [
    "nodes_pos_median_m",
    "nodes_pos_p95_m",
    "nodes_heading_rmse_deg",
    "rpe60s_nodes_trans_rmse_m",
    "inj_outage_med_at_30s_m",
    "inj_outage_med_at_60s_m",
    "inj_outage_med_at_120s_m",
    "nees_pos_mean",
]


def compare() -> None:
    """B vs A on the validation days (outages mode), per day and mean."""
    a = pd.read_csv(repo_root() / "reports" / "phase3_baseline" / "validation_runs.csv")
    a = a[(a["mode"] == "outages") & (a["config"] == load_parameters().filter.name)]
    b = pd.read_csv(OUT / "validation_runs_b.csv")
    b = b[b["mode"] == "outages"]
    rows = []
    for tag, df in (("A", a), ("B pilot", b)):
        cols = [m for m in METRICS if m in df.columns]
        for _, r in df.iterrows():
            rows.append({"method": tag, "date": r["date"], **{m: r[m] for m in cols}})
        rows.append({"method": tag, "date": "mean", **{m: df[m].mean() for m in cols}})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "comparison_outages.csv", index=False, float_format="%.4g")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = load_parameters()  # snapshot once
    if what in ("all", "cache"):
        cache(p)
    if what in ("all", "train"):
        np.save(RUNS / "delta.npy", train(p))
    if what in ("all", "eval"):
        evaluate(p, np.load(RUNS / "delta.npy"))
    if what in ("all", "compare"):
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
