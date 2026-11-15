import dataclasses
import datetime as dt
import warnings

import numpy as np
import pytest

from lase.data.reference import GraphNodes, GroundTruth, Reference
from lase.evaluation.estimate import NavEstimate, estimate_from_reference, reference_samples
from lase.evaluation.metrics import compute_errors, relative_pose_error, summarize
from lase.evaluation.outages import outage_curves
from lase.evaluation.report import evaluate_estimate, natural_outage_spans
from tests.unit.test_evaluation import _streams_with_gps

T0 = 1_000_000_000_000_000


def _ref(duration_s: float = 200.0) -> Reference:
    t = T0 + np.arange(0, int(duration_s * 1e6), 10_000, dtype=np.int64)
    s = (t - T0) * 1e-6
    p = np.stack([s * 1.0, np.zeros_like(s), np.zeros_like(s)], axis=1)  # 1 m/s north
    rpy = np.zeros((len(t), 3))
    gt = GroundTruth(t, p, rpy, np.zeros(len(t), dtype=np.uint16))
    idx = np.arange(0, len(t), 360, dtype=np.int64)
    nodes = GraphNodes(
        t[idx],
        np.tile(np.eye(6), (len(idx), 1, 1)),
        idx,
        np.zeros(len(idx), dtype=np.int64),
        np.ones(len(idx), dtype=bool),
    )
    return Reference(gt, nodes)


def test_evaluate_estimate_end_to_end() -> None:
    ref = _ref()
    est = estimate_from_reference(ref)
    est = dataclasses.replace(
        est, pos_ne=est.pos_ne + np.array([1.0, 0.0]), cov=np.tile(np.eye(5), (len(est.t_us), 1, 1))
    )
    spans = [(T0 + 50_000_000, T0 + 110_000_000)]
    rep = evaluate_estimate(est, ref, injected_spans=spans, natural_spans=spans)
    assert rep.nodes["pos_rmse_m"] == pytest.approx(1.0, abs=0.02)
    assert rep.dense["pos_median_m"] == pytest.approx(1.0, abs=0.02)
    assert rep.rpe[10.0].trans_rmse_m < 0.05
    assert rep.rpe_nodes[10.0].n > 0  # graph-node-anchored drift (nodes every 3.6 s here)
    assert rep.rpe_nodes[10.0].trans_rmse_m < 0.05
    assert rep.injected_outages is not None
    assert rep.nees_pos is not None
    assert rep.nees_pos.mean == pytest.approx(1.0, abs=0.05)  # error (1, 0), unit covariance
    row = rep.row()
    assert row["method"] == "reference"
    assert "inj_outage_med_at_30s_m" in row
    assert "rpe10s_nodes_trans_rmse_m" in row
    assert "rpe10s_dense_trans_rmse_m" in row


def test_natural_outage_spans() -> None:
    t = np.r_[np.arange(0, 10_000_001, 200_000), np.arange(20_000_000, 30_000_001, 200_000)].astype(
        np.int64
    )
    assert natural_outage_spans(_streams_with_gps(t)) == [(10_000_000, 20_000_000)]


def test_edge_cases_do_not_crash_or_warn() -> None:
    ref = _ref()
    empty = NavEstimate(
        "empty", np.zeros(0, dtype=np.int64), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    )
    samples = reference_samples(ref, "nodes")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = compute_errors(empty, samples)
        assert not e.associated.any()
        s = summarize(e)
        assert np.isnan(s["pos_rmse_m"])
        assert s["coverage"] == 0.0
        r = relative_pose_error(empty, samples, 10.0)
        assert r.n == 0
        assert np.isnan(r.trans_median_m)
        c = outage_curves(e, [(T0 + 10**12, T0 + 10**12 + 60_000_000)])
        assert np.isnan(c.errors_m).all()
        assert summarize([])["n"] == 0.0
    # estimate with NaN velocity/heading (position-only method)
    est = estimate_from_reference(ref)
    pos_only = dataclasses.replace(
        est, vel_ne=np.full_like(est.vel_ne, np.nan), heading=np.full_like(est.heading, np.nan)
    )
    s = summarize(compute_errors(pos_only, samples))
    assert s["pos_rmse_m"] < 0.05
    assert np.isnan(s["vel_rmse_mps"])
    # window outside the reference span
    out = reference_samples(ref, "dense", t_window=(T0 + 10**13, T0 + 10**13 + 1))
    assert len(out.t_us) == 0
    _ = dt
