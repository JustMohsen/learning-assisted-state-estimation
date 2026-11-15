"""GPS-outage injection and outage error-growth analysis."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lase.data.raw import F64, GpsRecord
from lase.data.session import SensorStreams
from lase.evaluation.metrics import ErrorSeries

Span = tuple[int, int]


def plan_outages(
    streams: SensorStreams,
    durations_s: Sequence[float] = (30.0, 60.0, 120.0),
    per_duration: int = 3,
    min_separation_s: float = 180.0,
    max_natural_gap_s: float = 2.0,
    seed: int = 0,
) -> list[Span]:
    """Choose non-overlapping outage windows where GPS was actually available.

    A window qualifies if valid GPS fixes have no gap over ``max_natural_gap_s`` inside
    it or in the ``min_separation_s`` before it (so the filter is GPS-aided beforehand).
    Windows are separated by at least ``min_separation_s`` for recovery.
    """
    rng = np.random.default_rng(seed)
    t = streams.gps.t_us[streams.gps.valid]
    if len(t) < 2:
        return []
    gaps = np.diff(t) > max_natural_gap_s * 1e6
    gap_starts, gap_ends = t[:-1][gaps], t[1:][gaps]
    chosen: list[Span] = []
    sep = int(min_separation_s * 1e6)
    for dur in sorted(durations_s, reverse=True):
        d = int(dur * 1e6)
        candidates = np.arange(t[0] + sep, t[-1] - d - sep, int(5e6), dtype=np.int64)
        rng.shuffle(candidates)
        n = 0
        for a in candidates:
            b = a + d
            if np.any((gap_ends > a - sep) & (gap_starts < b)):
                continue
            if any(a < cb + sep and b + sep > ca for ca, cb in chosen):
                continue
            chosen.append((int(a), int(b)))
            n += 1
            if n == per_duration:
                break
    return sorted(chosen)


def withhold_gps(streams: SensorStreams, spans: Sequence[Span]) -> SensorStreams:
    """Copy of ``streams`` with all GPS fixes inside ``spans`` removed."""
    t = streams.gps.t_us
    keep = np.ones(len(t), dtype=bool)
    for a, b in spans:
        keep &= ~((t >= a) & (t < b))
    g = streams.gps
    gps = GpsRecord(
        t_us=g.t_us[keep],
        flags=g.flags[keep],
        mode=g.mode[keep],
        num_sats=g.num_sats[keep],
        lat=g.lat[keep],
        lon=g.lon[keep],
        alt=g.alt[keep],
        track=g.track[keep],
        speed=g.speed[keep],
    )
    return dataclasses.replace(streams, gps=gps, gnss_ned=streams.gnss_ned[keep])


@dataclass(frozen=True)
class OutageCurves:
    """Position error vs time since outage start, one row per outage."""

    offsets_s: F64  # (K,)
    errors_m: F64  # (n_outages, K); NaN where unavailable
    durations_s: F64  # (n_outages,)
    final_error_m: F64  # error at the outage end

    def percentile(self, q: float) -> F64:
        """Per-offset percentile across outages (ignoring NaN; NaN where no outage is that long)."""
        out: F64 = np.full(self.errors_m.shape[1], np.nan)
        for k in range(self.errors_m.shape[1]):
            col = self.errors_m[:, k]
            col = col[np.isfinite(col)]
            if len(col):
                out[k] = np.percentile(col, q)
        return out


def outage_curves(
    errors: ErrorSeries,
    spans: Sequence[Span],
    offsets_s: Sequence[float] | None = None,
    sample_tolerance_s: float = 3.0,
) -> OutageCurves:
    """Sample horizontal error at fixed offsets after each outage start (nearest reference time)."""
    off = np.asarray(offsets_s if offsets_s is not None else [0, 5, 10, 20, 30, 45, 60, 90, 120])
    t = errors.t_us
    en = errors.pos_err_norm
    rows, finals, durs = [], [], []

    def sample(tq: int) -> float:
        k = int(np.searchsorted(t, tq))
        cands = [c for c in (k - 1, k) if 0 <= c < len(t)]
        if not cands:
            return float("nan")
        c = min(cands, key=lambda c: abs(int(t[c]) - tq))
        return float(en[c]) if abs(int(t[c]) - tq) <= sample_tolerance_s * 1e6 else float("nan")

    for a, b in spans:
        dur = (b - a) * 1e-6
        rows.append([sample(a + int(o * 1e6)) if o <= dur else np.nan for o in off])
        finals.append(sample(b))
        durs.append(dur)
    return OutageCurves(
        offsets_s=off.astype(np.float64),
        errors_m=np.asarray(rows, dtype=np.float64).reshape(len(rows), len(off)),
        durations_s=np.asarray(durs, dtype=np.float64),
        final_error_m=np.asarray(finals, dtype=np.float64),
    )
