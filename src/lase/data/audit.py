"""All-session data audit: rates, gaps, validity, GPS quality/outages, reference coverage."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from lase.data.raw import F64, I64, RowFlag
from lase.data.reference import Reference
from lase.data.session import SensorStreams


def spans_where(t_us: I64, mask: np.ndarray, min_duration_s: float) -> list[tuple[int, int]]:
    """Contiguous spans (start_us, end_us) where ``mask`` holds for at least ``min_duration_s``."""
    if len(t_us) == 0:
        return []
    m = mask.astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    out = []
    for a, b in zip(starts, ends, strict=True):
        if (t_us[b] - t_us[a]) * 1e-6 >= min_duration_s:
            out.append((int(t_us[a]), int(t_us[b])))
    return out


def gap_spans(t_us: I64, min_gap_s: float) -> list[tuple[int, int]]:
    """Intervals between consecutive samples longer than ``min_gap_s``."""
    if len(t_us) < 2:
        return []
    d = np.diff(t_us)
    i = np.flatnonzero(d > min_gap_s * 1e6)
    return [(int(t_us[k]), int(t_us[k + 1])) for k in i]


def _interval_stats(prefix: str, t_us: I64) -> dict[str, float]:
    d = np.diff(t_us) * 1e-3 if len(t_us) > 1 else np.array([np.nan])
    span = (t_us[-1] - t_us[0]) * 1e-6 if len(t_us) > 1 else np.nan
    return {
        f"{prefix}_n": float(len(t_us)),
        f"{prefix}_rate_hz": (len(t_us) - 1) / span if span else np.nan,
        f"{prefix}_dt_p50_ms": float(np.percentile(d, 50)),
        f"{prefix}_dt_p99_ms": float(np.percentile(d, 99)),
        f"{prefix}_dt_max_ms": float(np.max(d)),
    }


def _flag_frac(flags: np.ndarray, flag: RowFlag) -> float:
    return float(np.mean((flags & np.uint16(flag)) != 0)) if len(flags) else np.nan


def audit_session(
    streams: SensorStreams, ref: Reference, split: str
) -> tuple[dict[str, Any], dict[str, list[tuple[int, int]]]]:
    """Audit one session.

    Returns:
        A flat dict of summary statistics, and a dict of notable spans (GPS outages,
        stationary periods, input gaps) as (start_us, end_us) lists.
    """
    s = streams
    t_start = min(s.imu.t_us[0], s.fog.t_us[0], s.wheels.t_us[0], s.gps.t_us[0])
    t_end = max(s.imu.t_us[-1], s.fog.t_us[-1], s.wheels.t_us[-1], s.gps.t_us[-1])
    row: dict[str, Any] = {
        "date": s.date.isoformat(),
        "split": split,
        "duration_min": (t_end - t_start) * 1e-6 / 60,
        "start_utc": dt.datetime.fromtimestamp(t_start * 1e-6, tz=dt.UTC).isoformat(
            timespec="seconds"
        ),
    }
    for name, rec in [("imu", s.imu), ("fog", s.fog), ("wheel", s.wheels), ("gps", s.gps)]:
        row.update(_interval_stats(name, rec.t_us))
        row[f"{name}_invalid_frac"] = float(1 - rec.valid.mean())
        row[f"{name}_nonmono"] = int(((rec.flags & np.uint16(RowFlag.NON_MONOTONIC)) != 0).sum())
        row[f"{name}_dup_time"] = int(((rec.flags & np.uint16(RowFlag.DUPLICATE_TIME)) != 0).sum())

    # GPS quality and outages (no valid fix for > 5 s, including before first/after last fix)
    g = s.gps
    row["gps_mode2_frac"] = float(np.mean(g.mode == 2))
    row["gps_mode3_frac"] = float(np.mean(g.mode == 3))
    row["gps_nofix_frac"] = _flag_frac(g.flags, RowFlag.GPS_NO_FIX)
    row["gps_held_frac"] = _flag_frac(g.flags, RowFlag.GPS_HELD)
    row["gps_sats_nonzero_frac"] = float(np.mean(g.num_sats > 0))
    tv = g.t_us[g.valid]
    bounds = np.concatenate([[t_start], tv, [t_end]])
    outages = gap_spans(bounds, 5.0)
    row["gps_outages_n"] = len(outages)
    row["gps_outage_total_min"] = sum(b - a for a, b in outages) * 1e-6 / 60
    row["gps_outage_longest_s"] = max(((b - a) * 1e-6 for a, b in outages), default=0.0)

    # Stationary periods from wheels (both speeds ~0 for >= 5 s)
    w = s.wheels
    still = (np.abs(w.left) < 0.01) & (np.abs(w.right) < 0.01)
    stationary = spans_where(w.t_us, still, 5.0)
    row["stationary_n"] = len(stationary)
    row["stationary_total_min"] = sum(b - a for a, b in stationary) * 1e-6 / 60
    row["reverse_frac"] = float(np.mean((w.left + w.right) < -0.05))
    row["max_wheel_speed_mps"] = float(np.max(np.abs(np.concatenate([w.left, w.right]))))

    # Reference coverage
    gt = ref.gt
    tg = gt.t_us[gt.valid]
    row.update(_interval_stats("gt", tg))
    row["gt_invalid_rows"] = int((~gt.valid).sum())
    row["gt_start_after_sensors_s"] = (tg[0] - t_start) * 1e-6
    row["gt_end_before_sensors_s"] = (t_end - tg[-1]) * 1e-6
    row["gt_gaps_over_1s"] = len(gap_spans(tg, 1.0))
    tn = ref.nodes.t_us
    row["nodes_n"] = len(tn)
    row["node_dt_p50_s"] = float(np.median(np.diff(tn)) * 1e-6) if len(tn) > 1 else np.nan
    row["node_dt_max_s"] = float(np.max(np.diff(tn)) * 1e-6) if len(tn) > 1 else np.nan
    node_p: F64 = ref.node_p_w
    row["node_spacing_p50_m"] = (
        float(np.median(np.linalg.norm(np.diff(node_p[:, :2], axis=0), axis=1)))
        if len(tn) > 1
        else np.nan
    )
    row["node_max_offset_ms"] = float(np.max(np.abs(ref.nodes.gt_offset_us)) * 1e-3)
    row["node_pos_std_p50_m"] = float(
        np.median(np.sqrt(ref.nodes.cov[:, 0, 0] + ref.nodes.cov[:, 1, 1]))
    )

    spans = {
        "gps_outages": outages,
        "stationary": stationary,
        "imu_gaps_over_0.1s": gap_spans(s.imu.t_us, 0.1),
        "fog_gaps_over_0.1s": gap_spans(s.fog.t_us, 0.1),
        "wheel_gaps_over_0.2s": gap_spans(s.wheels.t_us, 0.2),
    }
    for k, v in spans.items():
        if k.endswith("s") and "gaps" in k:
            row[f"{k}_n"] = len(v)
    return row, spans
