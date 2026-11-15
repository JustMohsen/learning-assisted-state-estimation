"""Standard evaluation entry point: one call produces every Phase 2 metric for an estimate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lase.data.reference import Reference
from lase.data.session import SensorStreams
from lase.evaluation.consistency import ConsistencyResult, position_nees
from lase.evaluation.estimate import NavEstimate, reference_samples
from lase.evaluation.metrics import (
    ErrorSeries,
    RpeResult,
    compute_errors,
    relative_pose_error,
    summarize,
)
from lase.evaluation.outages import OutageCurves, Span, outage_curves


def natural_outage_spans(streams: SensorStreams, min_gap_s: float = 5.0) -> list[Span]:
    """Gaps between consecutive valid GPS fixes longer than ``min_gap_s`` (inside the session)."""
    t = streams.gps.t_us[streams.gps.valid]
    if len(t) < 2:
        return []
    d = np.diff(t)
    k = np.flatnonzero(d > min_gap_s * 1e6)
    return [(int(t[i]), int(t[i + 1])) for i in k]


@dataclass(frozen=True)
class EvaluationReport:
    """All standard metrics of one estimate on one session."""

    name: str
    nodes: dict[str, float]
    dense: dict[str, float]
    rpe: dict[
        float, RpeResult
    ]  # dense reference (odometry-interpolated between nodes; sensitivity)
    rpe_nodes: dict[float, RpeResult]  # graph nodes only (LiDAR/RTK-anchored; primary)
    injected_outages: OutageCurves | None
    natural_outages: OutageCurves | None
    nees_pos: ConsistencyResult | None
    dense_errors: ErrorSeries

    def row(self) -> dict[str, Any]:
        """Flat dictionary for tables (CSV)."""
        out: dict[str, Any] = {"method": self.name}
        out.update({f"nodes_{k}": v for k, v in self.nodes.items()})
        out.update({f"dense_{k}": v for k, v in self.dense.items()})
        for tag, res in (("nodes", self.rpe_nodes), ("dense", self.rpe)):
            for d, r in res.items():
                out[f"rpe{d:g}s_{tag}_trans_rmse_m"] = r.trans_rmse_m
                out[f"rpe{d:g}s_{tag}_heading_rmse_deg"] = r.heading_rmse_deg
        for tag, c in (("inj", self.injected_outages), ("nat", self.natural_outages)):
            if c is not None and len(c.errors_m):
                out[f"{tag}_outages_n"] = len(c.errors_m)
                for o, med in zip(c.offsets_s, c.percentile(50), strict=True):
                    out[f"{tag}_outage_med_at_{o:g}s_m"] = float(med)
                fin = c.final_error_m[np.isfinite(c.final_error_m)]
                out[f"{tag}_outage_final_med_m"] = float(np.median(fin)) if len(fin) else np.nan
        if self.nees_pos is not None:
            out["nees_pos_mean"] = self.nees_pos.mean
            out["nees_pos_cov95"] = self.nees_pos.coverage_95
        return out


def evaluate_estimate(
    est: NavEstimate,
    ref: Reference,
    *,
    t_window: tuple[int, int] | None = None,
    injected_spans: Sequence[Span] = (),
    natural_spans: Sequence[Span] = (),
    rpe_deltas_s: Sequence[float] = (10.0, 60.0),
    max_staleness_us: int = 200_000,
    dense_rate_hz: float = 10.0,
    curve_offsets_s: Sequence[float] | None = None,
    curve_sample_tolerance_s: float = 3.0,
    velocity_half_window_s: float = 0.05,
    max_speed_mps: float = 5.0,
) -> EvaluationReport:
    """Evaluate an estimate against the reference.

    Args:
        est: The method's output.
        ref: Session reference.
        t_window: Restrict scoring to this time window (e.g. after initialisation).
        injected_spans: GPS-withheld windows used when producing ``est`` (outage curves).
        natural_spans: Natural GPS outages (see :func:`natural_outage_spans`).
    """
    kw = {"velocity_half_window_s": velocity_half_window_s, "max_speed_mps": max_speed_mps}
    nodes_s = reference_samples(ref, "nodes", t_window=t_window, **kw)
    dense_s = reference_samples(ref, "dense", dense_rate_hz, t_window=t_window, **kw)
    e_nodes = compute_errors(est, nodes_s, max_staleness_us)
    e_dense = compute_errors(est, dense_s, max_staleness_us)
    nees = None
    if est.cov is not None and e_dense.associated.any():
        nees = position_nees(est, e_dense)
    return EvaluationReport(
        name=est.name,
        nodes=summarize(e_nodes),
        dense=summarize(e_dense),
        rpe={d: relative_pose_error(est, dense_s, d, max_staleness_us) for d in rpe_deltas_s},
        rpe_nodes={d: relative_pose_error(est, nodes_s, d, max_staleness_us) for d in rpe_deltas_s},
        injected_outages=(
            outage_curves(e_dense, injected_spans, curve_offsets_s, curve_sample_tolerance_s)
            if injected_spans
            else None
        ),
        natural_outages=(
            outage_curves(e_dense, natural_spans, curve_offsets_s, curve_sample_tolerance_s)
            if natural_spans
            else None
        ),
        nees_pos=nees,
        dense_errors=e_dense,
    )
