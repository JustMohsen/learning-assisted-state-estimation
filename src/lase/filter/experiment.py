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
    for name, log, dof in (
        ("wheel", res.wheel, 1),
        ("gnss", res.gnss, 2),
        ("nhc", res.nonholonomic, 2),
    ):
        if len(log["nis"]):
            r = nis(
                log["innovation"].reshape(len(log["nis"]), dof),
                log["S"].reshape(len(log["nis"]), dof, dof),
            )
            row[f"{name}_nis_mean"] = r.mean
            row[f"{name}_nis_median"] = float(np.median(log["nis"]))
            row[f"{name}_accept"] = float(log["accepted"].mean())
    row["gnss_recoveries"] = res.gnss_recoveries
    row["gnss_reacquisitions"] = res.gnss_reacquisitions
    row["final_wheel_scale"] = (
        float(res.biases["wheel_scale"][-1]) if len(res.biases["wheel_scale"]) else np.nan
    )
    row["minutes"] = (t_end - ic.t_us) / 60e6
    row["wall_s"] = res.wall_time_s
    row["us_per_step"] = 1e6 * res.wall_time_s / max(res.n_imu_steps, 1)
    if return_details:
        return row, rep, res.estimate, spans, natural
    return row


def _worker(args: tuple[Any, ...]) -> Any:
    import torch

    torch.set_num_threads(1)
    return run_session(*args)


def run_many(
    jobs: Sequence[tuple[Any, ...]],
    workers: int | None = None,
) -> list[Any]:
    """Run jobs in parallel processes (one thread each); results in job order."""
    n = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(_worker, jobs))


__all__ = ["replace", "run_many", "run_session"]
