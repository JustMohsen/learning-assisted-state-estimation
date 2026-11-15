"""Uncertainty consistency: NEES/NIS statistics and chi-square bounds.

NEES against the NCLT reference is only indicative: the reference is not independent
of the wheels/FOG and has its own uncertainty (see PROJECT_SCOPE.md, section 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from lase.data.raw import F64
from lase.evaluation.estimate import NavEstimate, associate
from lase.evaluation.metrics import ErrorSeries


def normalized_squared(err: F64, cov: F64) -> F64:
    """``e^T S^-1 e`` per sample for errors (N, d) and covariances (N, d, d)."""
    sol = np.linalg.solve(cov, err[..., None])[..., 0]
    out: F64 = np.einsum("nd,nd->n", err, sol)
    return out


@dataclass(frozen=True)
class ConsistencyResult:
    """Summary of a normalized-squared-error test (NEES or NIS)."""

    dof: int
    n: int
    mean: float  # should be close to dof
    coverage_95: float  # fraction inside the two-sided 95% chi-square interval (~0.95)
    frac_above_99: float  # fraction above the 99% quantile (~0.01)


def chi2_summary(values: F64, dof: int) -> ConsistencyResult:
    """Compare normalized squared errors with the chi-square distribution of ``dof``."""
    v = values[np.isfinite(values)]
    lo, hi = float(chi2.ppf(0.025, dof)), float(chi2.ppf(0.975, dof))
    return ConsistencyResult(
        dof=dof,
        n=len(v),
        mean=float(v.mean()) if len(v) else float("nan"),
        coverage_95=float(np.mean((v >= lo) & (v <= hi))) if len(v) else float("nan"),
        frac_above_99=float(np.mean(v > float(chi2.ppf(0.99, dof)))) if len(v) else float("nan"),
    )


def position_nees(est: NavEstimate, errors: ErrorSeries) -> ConsistencyResult:
    """Horizontal-position NEES (2 dof) using the estimate's covariance."""
    if est.cov is None:
        raise ValueError(f"Estimate '{est.name}' has no covariance.")
    idx, _ = associate(est, errors.t_us)
    m = errors.associated
    vals = normalized_squared(errors.pos_err[m], est.cov[idx[m]][:, :2, :2])
    return chi2_summary(vals, 2)


def nis(innovations: F64, S: F64) -> ConsistencyResult:
    """NIS statistics for update innovations (N, d) with innovation covariances (N, d, d)."""
    return chi2_summary(normalized_squared(innovations, S), innovations.shape[1])
