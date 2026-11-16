"""Phase 5 pilot: train method C (GRU noise multipliers) and compare with A and B.

    uv run python scripts/phase5_pilot.py [all|train|eval|compare]

Reuses the Phase 4 pilot's sessions, A snapshots (runs/phase4_pilot/) and windows.
1. train: Adam on the network; each step averages one chunk gradient per session.
2. eval: C runs through ``lase.filter.experiment.run_session`` (same path as A and B) on
   the validation days, with the network's outputs fed to ``run_filter`` as ``learned_fn``.
3. compare: outages mode, A vs B pilot vs C pilot.

Outputs: runs/phase5_pilot/ (network, log) and reports/phase5_pilot/. Test days untouched.
"""

from __future__ import annotations

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
from lase.parameters import Parameters, load_parameters
from lase.training.pilot_c import chunk_grad_c, grid, learned_fn_from, make_net
from lase.utils.paths import repo_root

CACHE = repo_root() / "runs" / "phase4_pilot"
RUNS = repo_root() / "runs" / "phase5_pilot"
OUT = repo_root() / "reports" / "phase5_pilot"
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


def pilot_dates(p: Parameters) -> list[Any]:
    train = sorted(load_split().dates(Split.TRAIN))
    n = min(p.training.pilot.n_sessions, len(train))
    idx = np.unique(np.linspace(0, len(train) - 1, n).round().astype(int))
    return [train[i] for i in idx]


def lr_at(step: int, p: Parameters) -> float:
    c, lo = p.training.pilot_c, p.training.optimizer.lr_final
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    frac = (step - c.warmup_steps) / max(c.steps - c.warmup_steps, 1)
    return lo + 0.5 * (c.lr - lo) * (1 + math.cos(math.pi * frac))


def train(p: Parameters) -> None:
    c, w = p.training.pilot_c, p.training.windows
    chunk_s = w.burn_in_s + w.loss_window_s
    snaps: dict[Any, list[Any]] = {}
    for d in pilot_dates(p):
        with (CACHE / f"snapshots_{d.isoformat()}.pkl").open("rb") as f:
            all_s = pickle.load(f)
        t_last = all_s[-1].t_us
        # first snapshot needs GRU warm-up history before it; last needs a full chunk after
        snaps[d] = [
            s
            for s in all_s
            if s.t_us + chunk_s * 1e6 < t_last and s.t_us - c.gru_warmup_s * 1e6 > all_s[0].t_us
        ]
    per = max(1, c.chunks_per_step // len(snaps))
    rng = np.random.default_rng(c.seed)
    net = make_net(p)
    opt = torch.optim.Adam(net.parameters(), lr=c.lr)
    pools = {d: ProcessPoolExecutor(max_workers=1) for d in snaps}
    RUNS.mkdir(parents=True, exist_ok=True)
    log = []
    print(f"train C: {len(snaps)} sessions, {per} chunk(s) per session per step", flush=True)
    try:
        for step in range(c.steps):
            t0 = time.time()
            state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
            futs = [
                pools[d].submit(chunk_grad_c, (d, ss[i], state, p))
                for d, ss in snaps.items()
                for i in rng.choice(len(ss), size=per, replace=False)
            ]
            res = [f.result() for f in futs]
            ok = [r for r in res if r["n"] > 0]
            opt.zero_grad()
            for k, prm in net.named_parameters():
                prm.grad = torch.as_tensor(np.mean([r["grads"][k] for r in ok], axis=0))
            gnorm = float(
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), p.training.optimizer.grad_clip_norm
                )
            )
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, p)
            opt.step()
            mults = np.array([r["mults"] for r in ok])
            row = {
                "step": step,
                "task_loss": float(np.mean([r["loss"] for r in ok])),
                "grad_norm": gnorm,
                "lr": lr_at(step, p),
                "wall_s": time.time() - t0,
                **{
                    f"mult_{k}_{s}": float(fn(mults[:, i]))
                    for i, k in enumerate(("gnss", "wheel", "nhc"))
                    for s, fn in (("med", np.median), ("min", np.min), ("max", np.max))
                },
            }
            log.append(row)
            pd.DataFrame(log).to_csv(RUNS / "train_log.csv", index=False)
            torch.save(net.state_dict(), RUNS / "net.pt")
            print(
                f"step {step:3d} loss {row['task_loss']:.4f} |g| {gnorm:.3f} mult med "
                f"gps {row['mult_gnss_med']:.2f} wheel {row['mult_wheel_med']:.2f} "
                f"nhc {row['mult_nhc_med']:.2f} ({row['wall_s']:.0f}s)",
                flush=True,
            )
    finally:
        for ex in pools.values():
            ex.shutdown()


def _eval_job(args: tuple[Any, ...]) -> Any:
    """Run one validation session with C's multipliers, through ``run_session``."""
    import lase.filter.experiment as experiment
    from lase.filter.runner import run_filter

    torch.set_num_threads(1)
    d, mode, p = args
    net = make_net(p)
    net.load_state_dict(torch.load(RUNS / "net.pt"))
    net.eval()

    def run_filter_c(streams: Any, ic: Any, cfg: Any, params: Any, **kw: Any) -> Any:
        # features from the streams the filter actually sees (GPS withheld in outages mode)
        t_end = kw.get("t_end_us") or int(streams.imu.t_us[-1])
        tg = grid(
            ic.t_us - int(p.training.pilot_c.gru_warmup_s * 1e6), t_end, p.training.pilot_c.grid_hz
        )
        with torch.no_grad():
            fn = learned_fn_from(net, streams, tg)
            return run_filter(streams, ic, cfg, params, learned_fn=fn, **kw)

    experiment.run_filter = run_filter_c  # type: ignore[attr-defined,assignment]  # same path as A
    cfg = p.filter.model_copy(update={"name": "pilot-c"})
    return experiment.run_session(d, cfg, mode, params=p)


def evaluate(p: Parameters) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dates = load_split().dates(Split.VALIDATION)
    jobs = [(d, m, p) for m in ("outages", "gps") for d in dates]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        rows = list(ex.map(_eval_job, jobs))
    pd.DataFrame(rows).assign(config="pilot-c").to_csv(OUT / "validation_runs_c.csv", index=False)
    print(f"wrote {OUT / 'validation_runs_c.csv'}", flush=True)


def compare() -> None:
    a = pd.read_csv(repo_root() / "reports" / "phase3_baseline" / "validation_runs.csv")
    a = a[a["config"] == load_parameters().filter.name]
    b = pd.read_csv(repo_root() / "reports" / "phase4_pilot" / "validation_runs_b.csv")
    c = pd.read_csv(OUT / "validation_runs_c.csv")
    rows = []
    for mode in ("outages", "gps"):
        for tag, df in (("A", a), ("B pilot", b), ("C pilot", c)):
            d = df[df["mode"] == mode]
            cols = [m for m in METRICS if m in d.columns]
            for _, r in d.iterrows():
                rows.append(
                    {"mode": mode, "method": tag, "date": r["date"], **{m: r[m] for m in cols}}
                )
    out = pd.DataFrame(rows).sort_values(["mode", "date", "method"])
    out.to_csv(OUT / "comparison.csv", index=False, float_format="%.4g")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = load_parameters()  # snapshot once
    if what in ("all", "train"):
        train(p)
    if what in ("all", "eval"):
        evaluate(p)
    if what in ("all", "compare"):
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
