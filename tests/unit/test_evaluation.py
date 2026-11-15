import dataclasses
import datetime as dt
import itertools

import numpy as np
import pytest

from lase.data.raw import FogRecord, GpsRecord, ImuRecord, WheelRecord
from lase.data.session import SensorStreams
from lase.evaluation.consistency import chi2_summary, normalized_squared
from lase.evaluation.estimate import NavEstimate, ReferenceSamples, associate
from lase.evaluation.metrics import compute_errors, relative_pose_error, summarize
from lase.evaluation.outages import outage_curves, plan_outages, withhold_gps

T = np.arange(0, 100_000_001, 100_000, dtype=np.int64)  # 100 s at 10 Hz


def _ref() -> ReferenceSamples:
    s = T * 1e-6
    psi = 0.02 * s
    pos = np.stack([10 * np.sin(psi), 10 * (1 - np.cos(psi))], axis=1)
    vel = np.gradient(pos, s, axis=0)
    return ReferenceSamples(T, pos, np.angle(np.exp(1j * psi)), vel, np.ones(len(T), dtype=bool))


def _est(
    ref: ReferenceSamples, offset: tuple[float, float] = (0, 0), dpsi: float = 0.0
) -> NavEstimate:
    return NavEstimate(
        "e", ref.t_us, ref.pos_ne + np.asarray(offset), ref.vel_ne, ref.heading + dpsi
    )


def test_associate_is_causal_and_respects_staleness() -> None:
    est = NavEstimate(
        "e",
        np.array([0, 100, 200], dtype=np.int64),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        np.zeros(3),
    )
    idx, ok = associate(
        est, np.array([-1, 0, 150, 199, 1_000], dtype=np.int64), max_staleness_us=500
    )
    assert idx[1:4].tolist() == [0, 1, 1]  # latest at or before, never a later one
    assert ok.tolist() == [False, True, True, True, False]


def test_errors_zero_for_perfect_and_offset_estimate() -> None:
    ref = _ref()
    assert summarize(compute_errors(_est(ref), ref))["pos_rmse_m"] == pytest.approx(0, abs=1e-12)
    s = summarize(compute_errors(_est(ref, (3.0, 4.0)), ref))
    assert s["pos_rmse_m"] == pytest.approx(5.0)
    assert s["coverage"] == 1.0


def test_heading_error_wraps() -> None:
    ref = _ref()
    est = dataclasses.replace(_est(ref), heading=ref.heading + 2 * np.pi + 0.01)
    assert summarize(compute_errors(est, ref))["heading_rmse_deg"] == pytest.approx(
        np.degrees(0.01), rel=1e-6
    )


def test_rpe_invariant_to_global_offset() -> None:
    ref = _ref()
    r = relative_pose_error(_est(ref, (100.0, -50.0)), ref, 10.0)
    assert r.trans_rmse_m == pytest.approx(0, abs=1e-9)
    assert r.n > 0


def test_chi2_summary_on_gaussian_samples() -> None:
    rng = np.random.default_rng(0)
    cov = np.tile(np.diag([4.0, 1.0]), (20000, 1, 1))
    err = rng.normal(0, 1, (20000, 2)) * np.array([2.0, 1.0])
    r = chi2_summary(normalized_squared(err, cov), 2)
    assert r.mean == pytest.approx(2.0, rel=0.03)
    assert r.coverage_95 == pytest.approx(0.95, abs=0.01)


def _streams_with_gps(t_gps: np.ndarray) -> SensorStreams:
    n = len(t_gps)
    z1 = np.zeros(1)
    f1 = np.zeros(1, dtype=np.uint16)
    t1 = np.zeros(1, dtype=np.int64)
    return SensorStreams(
        date=dt.date(2013, 1, 10),
        imu=ImuRecord(t1, f1, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))),
        fog=FogRecord(t1, f1, z1),
        wheels=WheelRecord(t1, f1, z1, z1),
        gps=GpsRecord(
            t_gps,
            np.zeros(n, dtype=np.uint16),
            np.full(n, 3),
            np.zeros(n, dtype=np.int64),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        ),
        fog_angle=z1,
        gnss_ned=np.zeros((n, 3)),
    )


def test_plan_outages_avoids_natural_gaps_and_withhold_removes() -> None:
    t = np.arange(0, 3_600_000_001, 200_000, dtype=np.int64)  # 1 h at 5 Hz
    gap = (t > 1_800_000_000) & (t < 1_900_000_000)  # natural 100 s gap
    s = _streams_with_gps(t[~gap])
    spans = plan_outages(s, durations_s=(60.0,), per_duration=5, min_separation_s=120.0, seed=1)
    assert spans
    for a, b in spans:
        assert b - a == 60_000_000
        assert b + 120_000_000 <= 1_800_000_000 or a - 120_000_000 >= 1_900_000_000
    for (_, b1), (a2, _) in itertools.pairwise(spans):
        assert a2 - b1 >= 120_000_000
    w = withhold_gps(s, spans)
    for a, b in spans:
        assert not np.any((w.gps.t_us >= a) & (w.gps.t_us < b))
    assert len(w.gnss_ned) == len(w.gps)


def test_outage_curves_sampling() -> None:
    ref = _ref()
    est = dataclasses.replace(
        _est(ref), pos_ne=ref.pos_ne + np.stack([T * 1e-6, np.zeros(len(T))], axis=1)
    )
    c = outage_curves(
        compute_errors(est, ref), [(10_000_000, 40_000_000)], offsets_s=[0, 10, 30, 60]
    )
    np.testing.assert_allclose(c.errors_m[0, :3], [10, 20, 40], atol=1e-6)
    assert np.isnan(c.errors_m[0, 3])  # beyond the outage duration
    assert c.final_error_m[0] == pytest.approx(40.0)
