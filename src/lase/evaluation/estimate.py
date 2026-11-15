"""Estimator output format and causal association with reference times."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lase.data.raw import F64, I64, B
from lase.data.reference import Reference
from lase.geometry.euler import wrap_angle

COV_ORDER = ("north", "east", "v_north", "v_east", "heading")


@dataclass(frozen=True)
class NavEstimate:
    """Planar navigation output of a method, at the body reference point (axle centre).

    Missing quantities are NaN (e.g. a position-only baseline has NaN velocity/heading).
    ``cov`` is (N, 5, 5) in the order of :data:`COV_ORDER`, or None.
    """

    name: str
    t_us: I64  # strictly increasing output times
    pos_ne: F64  # (N, 2)
    vel_ne: F64  # (N, 2)
    heading: F64  # (N,), wrapped to (-pi, pi]
    cov: F64 | None = None

    def __len__(self) -> int:
        return len(self.t_us)


def associate(est: NavEstimate, t_query: I64, max_staleness_us: int = 200_000) -> tuple[I64, B]:
    """Causal association: latest estimate at or before each query time.

    Returns:
        Indices into ``est`` and a mask (False where no estimate is recent enough).
    """
    if len(est.t_us) == 0:
        return np.zeros(len(t_query), dtype=np.int64), np.zeros(len(t_query), dtype=bool)
    idx = np.searchsorted(est.t_us, t_query, side="right") - 1
    ok = idx >= 0
    safe = np.clip(idx, 0, len(est.t_us) - 1)
    ok &= (t_query - est.t_us[safe]) <= max_staleness_us
    return safe.astype(np.int64), ok


@dataclass(frozen=True)
class ReferenceSamples:
    """Reference values at evaluation times."""

    t_us: I64
    pos_ne: F64
    heading: F64
    vel_ne: F64
    vel_valid: B


def reference_samples(
    ref: Reference,
    which: Literal["nodes", "dense"] = "nodes",
    dense_rate_hz: float = 10.0,
    t_window: tuple[int, int] | None = None,
) -> ReferenceSamples:
    """Reference at graph nodes (primary) or on a regular dense grid (sensitivity).

    Graph nodes use the matched dense pose (Phase 1). Velocity is the offline derived
    reference velocity (central differences, masked where unreliable).
    """
    m = ref.gt.valid
    if which == "nodes":
        nv = ref.nodes.valid
        t = ref.nodes.t_us[nv]
        p = ref.node_p_w[nv]
        yaw = ref.node_rpy[nv, 2]
    else:
        tg = ref.gt.t_us[m]
        t = np.arange(tg[0], tg[-1], round(1e6 / dense_rate_hz), dtype=np.int64)
        p, rpy = ref.pose_at(t)
        yaw = rpy[:, 2]
    if t_window is not None:
        keep = (t >= t_window[0]) & (t < t_window[1])
        t, p, yaw = t[keep], p[keep], yaw[keep]
    v, vmask = ref.velocity_w(t)
    return ReferenceSamples(
        t_us=t, pos_ne=p[:, :2], heading=wrap_angle(yaw), vel_ne=v[:, :2], vel_valid=vmask
    )


def estimate_from_reference(ref: Reference, rate_hz: float = 50.0) -> NavEstimate:
    """The reference itself as an estimate (sanity check: errors should be ~0 at nodes)."""
    m = ref.gt.valid
    tg = ref.gt.t_us[m]
    t = np.arange(tg[0], tg[-1], round(1e6 / rate_hz), dtype=np.int64)
    t = np.union1d(t, ref.nodes.t_us[ref.nodes.valid] + ref.nodes.gt_offset_us[ref.nodes.valid])
    p, rpy = ref.pose_at(t)
    v, _ = ref.velocity_w(t)
    return NavEstimate("reference", t, p[:, :2], v[:, :2], wrap_angle(rpy[:, 2]))
