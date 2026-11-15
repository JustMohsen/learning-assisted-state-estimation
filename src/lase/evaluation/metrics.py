"""Accuracy metrics: absolute errors at reference times and planar relative pose error."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lase.data.raw import F64, I64, B
from lase.evaluation.estimate import NavEstimate, ReferenceSamples, associate
from lase.geometry.euler import wrap_angle


@dataclass(frozen=True)
class ErrorSeries:
    """Per-sample errors (estimate minus reference) at reference times."""

    t_us: I64
    pos_err: F64  # (N, 2) north, east
    heading_err: F64  # (N,) wrapped
    vel_err: F64  # (N, 2)
    associated: B  # an estimate was available (causal, not stale)
    vel_valid: B  # reference velocity valid and estimate velocity finite

    @property
    def pos_err_norm(self) -> F64:
        """Horizontal position error magnitude (NaN where not associated)."""
        out: F64 = np.where(self.associated, np.linalg.norm(self.pos_err, axis=1), np.nan)
        return out


def compute_errors(
    est: NavEstimate, ref: ReferenceSamples, max_staleness_us: int = 200_000
) -> ErrorSeries:
    """Errors of ``est`` at the reference sample times (causal association)."""
    idx, ok = associate(est, ref.t_us, max_staleness_us)
    if len(est.t_us) == 0:
        n = len(ref.t_us)
        nan2, nan1 = np.full((n, 2), np.nan), np.full(n, np.nan)
        return ErrorSeries(ref.t_us, nan2, nan1, nan2.copy(), ok, np.zeros(n, dtype=bool))
    pos = est.pos_ne[idx] - ref.pos_ne
    head = wrap_angle(est.heading[idx] - ref.heading)
    vel = est.vel_ne[idx] - ref.vel_ne
    vel_ok = ok & ref.vel_valid & np.isfinite(vel).all(1)
    return ErrorSeries(ref.t_us, pos, head, vel, ok, vel_ok)


def _rmse(x: F64) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def summarize(errors: ErrorSeries | Sequence[ErrorSeries]) -> dict[str, float]:
    """Summary statistics; a sequence of series is pooled sample-wise (e.g. across days)."""
    es = [errors] if isinstance(errors, ErrorSeries) else list(errors)
    if not es or sum(len(e.t_us) for e in es) == 0:
        nan = float("nan")
        keys = ("pos_rmse_m", "pos_mean_m", "pos_median_m", "pos_p95_m", "pos_max_m")
        return {
            "n": 0.0,
            "coverage": nan,
            **dict.fromkeys(keys, nan),
            "heading_rmse_deg": nan,
            "vel_rmse_mps": nan,
        }
    a = np.concatenate([e.associated for e in es])
    pn = np.concatenate([e.pos_err_norm for e in es])[a]
    h = np.concatenate([e.heading_err for e in es])[a]
    vv = np.concatenate([e.vel_valid for e in es])
    v = np.concatenate([e.vel_err for e in es])[vv]
    vn = np.linalg.norm(v, axis=1) if len(v) else np.array([np.nan])
    has_pos = bool(np.isfinite(pn).any())
    return {
        "n": float(len(a)),
        "coverage": float(a.mean()) if len(a) else float("nan"),
        "pos_rmse_m": _rmse(pn),
        "pos_mean_m": float(np.nanmean(pn)) if has_pos else float("nan"),
        "pos_median_m": float(np.nanmedian(pn)) if has_pos else float("nan"),
        "pos_p95_m": float(np.nanpercentile(pn, 95)) if has_pos else float("nan"),
        "pos_max_m": float(np.nanmax(pn)) if has_pos else float("nan"),
        "heading_rmse_deg": float(np.degrees(_rmse(h))),
        "vel_rmse_mps": _rmse(vn),
    }


@dataclass(frozen=True)
class RpeResult:
    """Planar relative pose error over a fixed time interval."""

    delta_s: float
    n: int
    trans_rmse_m: float
    trans_median_m: float
    heading_rmse_deg: float


def relative_pose_error(
    est: NavEstimate, ref: ReferenceSamples, delta_s: float, max_staleness_us: int = 200_000
) -> RpeResult:
    """Compare displacement over ``delta_s`` expressed in the start heading frame.

    Pairs are formed from reference sample times ``t`` and the first sample at or after
    ``t + delta_s``. Estimate values use causal association at both times.
    """
    if len(est.t_us) == 0 or len(ref.t_us) == 0:
        nan = float("nan")
        return RpeResult(
            delta_s=delta_s, n=0, trans_rmse_m=nan, trans_median_m=nan, heading_rmse_deg=nan
        )
    t = ref.t_us
    j = np.searchsorted(t, t + int(delta_s * 1e6), side="left")
    ok_pair = j < len(t)
    i = np.flatnonzero(ok_pair)
    j = j[ok_pair]
    idx, ok = associate(est, t, max_staleness_us)
    good = ok[i] & ok[j]
    i, j = i[good], j[good]

    def rel(p: F64, psi: F64, a: I64, b: I64) -> tuple[F64, F64]:
        d = p[b] - p[a]
        c, s = np.cos(psi[a]), np.sin(psi[a])
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        return local, wrap_angle(psi[b] - psi[a])

    d_ref, h_ref = rel(ref.pos_ne, ref.heading, i, j)
    d_est, h_est = rel(est.pos_ne[idx], est.heading[idx], i, j)
    te = np.linalg.norm(d_est - d_ref, axis=1)
    he = wrap_angle(h_est - h_ref)
    finite = te[np.isfinite(te)]
    return RpeResult(
        delta_s=delta_s,
        n=len(te),
        trans_rmse_m=_rmse(te),
        trans_median_m=float(np.median(finite)) if len(finite) else float("nan"),
        heading_rmse_deg=float(np.degrees(_rmse(he))),
    )
