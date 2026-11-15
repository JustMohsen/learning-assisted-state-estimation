"""Run the baseline filter on real sessions and score it with the Phase 2 harness."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from lase.data.calibration import load_calibration
from lase.data.session import load_inputs, load_reference_for
from lase.evaluation.consistency import nis
from lase.evaluation.outages import plan_outages, withhold_gps
from lase.evaluation.report import evaluate_estimate, natural_outage_spans
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.runner import initial_condition_from_reference, run_filter
from lase.parameters import Parameters, load_parameters

Mode = Literal["gps", "outages", "nogps"]


def run_session(
    date: dt.date,
    cfg: FilterConfig,
    mode: Mode = "gps",
    max_minutes: float | None = None,
    start_offset_min: float = 0.0,
    return_details: bool = False,
    params: Parameters | None = None,
) -> Any:
    """Run and evaluate one session.

    Modes: ``gps`` (all sensors), ``outages`` (GPS withheld in the seeded injected windows),
    ``nogps`` (no GPS updates at all). Optionally restrict to a segment
    ``[start + offset, start + offset + max_minutes]``. Data and evaluation settings come
    from ``params`` (default: ``configs/parameters.yaml``); the filter from ``cfg``.
    """
    P = params or load_parameters()
    ev, op = P.evaluation, P.evaluation.outages
    cal = load_calibration()
    streams = load_inputs(date, cal, P.data.limits)
    ref = load_reference_for(date, P.data.node_match_tolerance_s)
    filt_params = FilterParams.build(cal, cfg)
    t_ref0 = int(ref.gt.t_us[ref.gt.valid][0])
    ic = initial_condition_from_reference(
        ref, t_ref0 + int((cfg.init.start_offset_s + start_offset_min * 60) * 1e6)
    )
    t_end = int(streams.imu.t_us[-1])
    if max_minutes is not None:
        t_end = min(t_end, ic.t_us + int(max_minutes * 60e6))
    earliest = ic.t_us + op.earliest_start_after_init_s * 1e6
    spans: list[tuple[int, int]] = []
    run_cfg = cfg
    s = streams
    if mode == "outages":
        planned = plan_outages(
            streams,
            durations_s=op.durations_s,
            per_duration=op.per_duration,
            min_separation_s=op.min_separation_s,
            max_natural_gap_s=op.max_natural_gap_s,
            seed=int(date.strftime("%Y%m%d")),
        )
        spans = [(a, b) for a, b in planned if a > earliest and b < t_end]
        s = withhold_gps(streams, spans)
    elif mode == "nogps":
        run_cfg = cfg.model_copy(update={"gnss": cfg.gnss.model_copy(update={"enabled": False})})
    res = run_filter(s, ic, run_cfg, filt_params, t_end_us=t_end)
    natural = [
        (a, b)
        for a, b in natural_outage_spans(streams, op.natural_min_gap_s)
        if a > earliest and b < t_end
    ]
    rep = evaluate_estimate(
        res.estimate,
        ref,
        t_window=(ic.t_us + int(ev.score_start_after_init_s * 1e6), t_end),
        injected_spans=spans,
        natural_spans=natural if mode != "nogps" else (),
        rpe_deltas_s=ev.rpe_deltas_s,
        max_staleness_us=int(ev.max_staleness_s * 1e6),
        dense_rate_hz=ev.dense_rate_hz,
        curve_offsets_s=op.curve_offsets_s,
        curve_sample_tolerance_s=op.curve_sample_tolerance_s,
        velocity_half_window_s=P.data.velocity_half_window_s,
        max_speed_mps=P.data.max_reference_speed_mps,
    )
    row: dict[str, Any] = {"date": date.isoformat(), "mode": mode, **rep.row()}
    for name, log in (("wheel", res.wheel), ("gnss", res.gnss), ("nhc", res.nonholonomic)):
        n = len(log["nis"])
        if n:
            dof = log["innovation"].reshape(n, -1).shape[1]  # measurement dimension from the data
            r = nis(log["innovation"].reshape(n, dof), log["S"].reshape(n, dof, dof))
            row[f"{name}_nis_mean"] = r.mean
            row[f"{name}_nis_median"] = float(np.median(log["nis"]))
            row[f"{name}_accept"] = float(log["accepted"].mean())
    row["gnss_recoveries"] = res.gnss_recoveries
    row["gnss_reacquisitions"] = res.gnss_reacquisitions
    row["final_wheel_scale"] = (
        float(res.biases["wheel_scale"][-1]) if len(res.biases["wheel_scale"]) else np.nan
    )
    row["final_wheel_scale_sigma"] = float(np.sqrt(res.final_P[0, 16, 16]))
    row["minutes"] = (t_end - ic.t_us) / 60e6
    row["wall_s"] = res.wall_time_s
    row["us_per_step"] = 1e6 * res.wall_time_s / max(res.n_imu_steps, 1)
    if return_details:
        return row, rep, res.estimate, spans, natural
    return row


def run_chain(
    dates: Sequence[dt.date],
    cfg: FilterConfig,
    mode: Mode = "gps",
    max_minutes: float | None = None,
    params: Parameters | None = None,
    out_csv: str | None = None,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Run sessions in date order, carrying the wheel-scale estimate (``cfg.wheel_scale_chain``).

    If ``out_csv`` is given, each session's row is appended as soon as it finishes, so a
    crash never loses completed sessions.
    """
    import pandas as pd

    params = params or load_parameters()  # snapshot once; never re-read during the chain
    ch = cfg.wheel_scale_chain
    value, sigma, prev = ch.first_value, ch.first_sigma, None
    rows = []
    for d in sorted(dates):
        if prev is not None:
            days = (d - prev).days
            sigma = float(
                np.sqrt(sigma**2 + ch.sigma_floor**2 + (ch.sigma_growth_per_day * days) ** 2)
            )
        c = cfg.model_copy(
            update={
                "wheel": cfg.wheel.model_copy(update={"speed_scale": value}),
                "prior": cfg.prior.model_copy(update={"wheel_scale": sigma}),
            }
        )
        row = run_session(d, c, mode, max_minutes, params=params)
        row["init_wheel_scale"], row["init_wheel_scale_sigma"] = value, sigma
        row["variant"] = tag
        rows.append(row)
        if out_csv is not None:
            pd.DataFrame([row]).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv), index=False
            )
        value, sigma, prev = row["final_wheel_scale"], row["final_wheel_scale_sigma"], d
    return rows


def _chain_worker(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    return run_chain(*args)


def _worker(args: tuple[Any, ...]) -> Any:
    import torch

    torch.set_num_threads(1)
    return run_session(*args)


def run_many(
    jobs: Sequence[tuple[Any, ...]],
    workers: int | None = None,
    params: Parameters | None = None,
) -> list[Any]:
    """Run ``run_session`` jobs in parallel processes (one thread each); results in job order.

    ``configs/parameters.yaml`` is loaded **once here** (or ``params`` is used) and passed to
    every job, so editing the file while jobs run cannot change or break them.
    """
    snapshot = params or load_parameters()
    full = []
    for job in jobs:
        j = list(job) + [None] * (6 - len(job))  # date, cfg, mode, max_minutes, offset, details
        j[3] = j[3] if len(job) > 3 else None
        j[4] = j[4] if len(job) > 4 and j[4] is not None else 0.0
        j[5] = bool(j[5]) if len(job) > 5 else False
        full.append((*j[:6], snapshot))
    n = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(_worker, full))


__all__ = ["replace", "run_many", "run_session"]
