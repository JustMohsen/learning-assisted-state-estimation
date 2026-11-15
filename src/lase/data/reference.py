"""Reference ("ground truth") trajectory: dense poses and SLAM graph nodes.

Evaluation- and training-side only. The estimator's event stream never exposes this
data. Dense poses (~150 Hz) are odometry-interpolated between SLAM graph nodes
(about every 8 m); graph-node poses are the most trustworthy, and their timestamps
come from the covariance file (official ``read_ground_truth.py``).

Poses are body-frame poses (axle centre; forward-right-down) in the dataset's local
NED frame: position ``p_w`` and NCLT Euler angles (roll, pitch, yaw), where
``C_wb = Rz(yaw) Ry(pitch) Rx(roll)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lase.data.conversions import interp_at
from lase.data.raw import F64, I64, U16, B, RowFlag, _flag, _time_flags, read_numeric_csv
from lase.geometry.euler import euler_zyx_to_matrix, wrap_angle

INVALID_REF = RowFlag.NON_MONOTONIC | RowFlag.NON_FINITE | RowFlag.DUPLICATE_TIME


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Dense reference poses."""

    t_us: I64
    p_w: F64  # (N, 3) body origin, NED (m)
    rpy: F64  # (N, 3) roll, pitch, yaw (rad)
    flags: U16

    @property
    def valid(self) -> B:
        """Rows usable as reference (finite, strictly increasing time)."""
        v: B = (self.flags & np.uint16(INVALID_REF)) == 0
        return v

    def __len__(self) -> int:
        return len(self.t_us)


@dataclass(frozen=True, slots=True)
class GraphNodes:
    """SLAM graph nodes: timestamps, marginal covariances and matched dense-pose indices."""

    t_us: I64
    cov: F64  # (M, 6, 6) over [x, y, z, roll, pitch, yaw]
    gt_index: I64  # nearest valid dense ground-truth row
    gt_offset_us: I64  # dense-row time minus node time
    valid: B  # node has a dense ground-truth row within the matching tolerance

    def __len__(self) -> int:
        return len(self.t_us)


def load_ground_truth(path: Path) -> GroundTruth:
    """Load ``groundtruth_<date>.csv``: utime, x, y, z, roll, pitch, yaw."""
    t, v = read_numeric_csv(path, 7)
    flags = _time_flags(t)
    _flag(flags, ~np.isfinite(v).all(1), RowFlag.NON_FINITE)
    return GroundTruth(t_us=t, p_w=v[:, 0:3].copy(), rpy=v[:, 3:6].copy(), flags=flags)


def _upper_to_full(u: F64) -> F64:
    """Row-major upper triangle (M, 21) to symmetric (M, 6, 6)."""
    iu = np.triu_indices(6)
    full = np.zeros((u.shape[0], 6, 6))
    full[:, iu[0], iu[1]] = u
    full[:, iu[1], iu[0]] = u
    return full


def load_graph_nodes(path: Path, gt: GroundTruth, max_offset_us: int = 50_000) -> GraphNodes:
    """Load ``cov_<date>.csv`` (utime + 21 upper-triangular covariance entries) and match to ``gt``.

    Nodes without a valid dense ground-truth row within ``max_offset_us`` are marked
    invalid. This happens in truncated releases (e.g. 2012-04-29, 2012-06-15), whose
    covariance files list nodes beyond the end of the released ground truth.
    """
    t, v = read_numeric_csv(path, 22)
    valid_idx = np.flatnonzero(gt.valid)
    tv = gt.t_us[valid_idx]
    j = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
    nearer_left = (t - tv[j - 1]) <= (tv[j] - t)
    k = np.where(nearer_left, j - 1, j)
    idx = valid_idx[k]
    offset = gt.t_us[idx] - t
    return GraphNodes(
        t_us=t,
        cov=_upper_to_full(v),
        gt_index=idx,
        gt_offset_us=offset,
        valid=np.abs(offset) <= max_offset_us,
    )


@dataclass(frozen=True, slots=True)
class Reference:
    """Dense poses plus graph nodes for one session, with offline derived quantities."""

    gt: GroundTruth
    nodes: GraphNodes

    @property
    def node_p_w(self) -> F64:
        """Graph-node positions (M, 3); use together with ``nodes.valid``."""
        out: F64 = self.gt.p_w[self.nodes.gt_index]
        return out

    @property
    def node_rpy(self) -> F64:
        """Graph-node Euler angles (M, 3); use together with ``nodes.valid``."""
        out: F64 = self.gt.rpy[self.nodes.gt_index]
        return out

    def pose_at(self, t_query: I64) -> tuple[F64, F64]:
        """*Offline* interpolated position (Q, 3) and Euler angles (Q, 3); NaN outside the span."""
        m = self.gt.valid
        t = self.gt.t_us[m]
        p = interp_at(t, self.gt.p_w[m], t_query)
        rpy = self.gt.rpy[m].copy()
        rpy[:, 2] = np.unwrap(rpy[:, 2])
        out = interp_at(t, rpy, t_query)
        out[:, 2] = wrap_angle(out[:, 2])
        return p, out

    def velocity_w(
        self, t_query: I64, half_window_us: int = 50_000, max_speed_mps: float = 5.0
    ) -> tuple[F64, B]:
        """*Offline* derived body-origin velocity in NED by central differences.

        Args:
            t_query: Query times (us).
            half_window_us: Half-width ``h`` of the difference ``(p(t+h) - p(t-h)) / 2h``.
            max_speed_mps: Speeds above this are masked as implausible.

        Returns:
            Velocity (Q, 3) and a validity mask (Q,). Invalid where the window leaves the
            valid reference span, crosses a reference gap longer than ``4h``, or is implausible.
        """
        m = self.gt.valid
        t = self.gt.t_us[m]
        tq = np.asarray(t_query, dtype=np.int64)
        p_plus = interp_at(t, self.gt.p_w[m], tq + half_window_us)
        p_minus = interp_at(t, self.gt.p_w[m], tq - half_window_us)
        v = (p_plus - p_minus) / (2 * half_window_us * 1e-6)
        # largest sample gap inside the window
        lo = np.searchsorted(t, tq - half_window_us)
        hi = np.searchsorted(t, tq + half_window_us)
        gaps = np.diff(t)
        cummax_ok = np.ones(len(tq), dtype=bool)
        for i in np.flatnonzero((hi - lo) >= 1):
            a, b = max(lo[i] - 1, 0), min(hi[i], len(gaps))
            if b > a and gaps[a:b].max() > 4 * half_window_us:
                cummax_ok[i] = False
        speed = np.linalg.norm(v[:, :2], axis=1)
        mask: B = np.isfinite(v).all(1) & cummax_ok & (speed <= max_speed_mps)
        return v, mask

    def forward_speed(self, t_query: I64, half_window_us: int = 50_000) -> tuple[F64, B]:
        """*Offline* derived forward speed in the body frame (m/s) and validity mask."""
        v_w, mask = self.velocity_w(t_query, half_window_us)
        _, rpy = self.pose_at(t_query)
        C_wb = euler_zyx_to_matrix(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        v_b = np.einsum("qji,qj->qi", C_wb, v_w)  # C_wb^T v_w
        fwd: F64 = v_b[:, 0]
        return fwd, mask & np.isfinite(fwd)


def load_reference(gt_path: Path, cov_path: Path) -> Reference:
    """Load dense ground truth and graph nodes for one session."""
    gt = load_ground_truth(gt_path)
    return Reference(gt=gt, nodes=load_graph_nodes(cov_path, gt))
