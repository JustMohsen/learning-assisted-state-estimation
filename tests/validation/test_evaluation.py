"""Evaluation-harness cross-checks on real data (E1: evo agreement; E2: sanity)."""

import datetime as dt

import numpy as np
import pytest

from lase.data.session import extracted_dir, load_diagnostics, load_reference_for
from lase.evaluation.baselines import odometry_baseline
from lase.evaluation.estimate import associate, estimate_from_reference, reference_samples
from lase.evaluation.metrics import compute_errors, summarize

pytestmark = pytest.mark.validation
DATE = dt.date(2012, 1, 22)


def test_e1_position_rmse_matches_evo() -> None:
    if not (extracted_dir(DATE) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")
    from evo.core import metrics
    from evo.core.trajectory import PoseTrajectory3D

    ref = load_reference_for(DATE)
    nodes = reference_samples(ref, "nodes")
    est = odometry_baseline(load_diagnostics(DATE).odometry_100hz, ref)
    ours = summarize(compute_errors(est, nodes))["pos_rmse_m"]
    idx, ok = associate(est, nodes.t_us)

    def traj(p: np.ndarray, psi: np.ndarray, t: np.ndarray) -> PoseTrajectory3D:
        z = np.zeros(len(p))
        q = np.c_[np.cos(psi / 2), z, z, np.sin(psi / 2)]
        return PoseTrajectory3D(
            positions_xyz=np.c_[p, z], orientations_quat_wxyz=q, timestamps=t * 1e-6
        )

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data(
        (
            traj(nodes.pos_ne[ok], nodes.heading[ok], nodes.t_us[ok]),
            traj(est.pos_ne[idx[ok]], est.heading[idx[ok]], nodes.t_us[ok]),
        )
    )
    assert ours == pytest.approx(ape.get_statistic(metrics.StatisticsType.rmse), abs=1e-6)


def test_e2_reference_scores_itself_near_zero() -> None:
    if not (extracted_dir(DATE) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")
    ref = load_reference_for(DATE)
    s = summarize(compute_errors(estimate_from_reference(ref), reference_samples(ref, "nodes")))
    assert s["pos_rmse_m"] < 0.02
    assert s["heading_rmse_deg"] < 0.1
