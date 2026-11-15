"""Phase 1 cross-sensor validation (V1-V5) on downloaded data.

Run explicitly: uv run pytest tests/validation
Skips if the NCLT data has not been downloaded.
"""

import datetime as dt

import numpy as np
import pytest

from lase.data.calibration import load_calibration
from lase.data.crosscheck import check_gps, check_track_width, check_wheel_speed, check_yaw_rate
from lase.data.session import extracted_dir, load_inputs, load_reference_for
from lase.data.splits import load_split

pytestmark = pytest.mark.validation

# Bring-up sessions plus others from different seasons and splits.
SAMPLE = [dt.date(2013, 1, 10), dt.date(2012, 1, 22), dt.date(2012, 5, 26), dt.date(2012, 11, 4)]


def _have(d: dt.date) -> bool:
    return (extracted_dir(d) / "ms25.csv").exists()


@pytest.fixture(scope="module", params=SAMPLE, ids=str)
def session(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    d = request.param
    if not _have(d):
        pytest.skip(f"NCLT data for {d} not downloaded")
    return load_inputs(d), load_reference_for(d)


def test_v1_gps_frame(session) -> None:  # type: ignore[no-untyped-def]
    s, ref = session
    r = check_gps(s, ref, load_calibration())
    assert r.median_err_m < 6.0
    assert abs(r.rotation_deg) < 1.0
    assert abs(r.scale_minus_1) < 0.01
    assert r.inlier_frac > 0.8


def test_v2_yaw_timing_sign_scale(session) -> None:  # type: ignore[no-untyped-def]
    s, ref = session
    sign = load_calibration().fog_convention.angle_sign_to_ned_yaw
    r = check_yaw_rate(s, ref)
    assert abs(r.fog_lag_ms) < 20
    assert abs(r.imu_lag_ms) < 20
    assert abs(r.fog_corr) > 0.9
    assert r.imu_corr > 0.9
    assert sign * r.fog_scale == pytest.approx(1.0, abs=0.03)
    assert r.imu_scale == pytest.approx(1.0, abs=0.03)


def test_v3_wheel_speed(session) -> None:  # type: ignore[no-untyped-def]
    s, ref = session
    r = check_wheel_speed(s, ref)
    assert 0.95 < r.scale < 1.15
    assert r.median_abs_residual_mps < 0.2


def test_v4_track_width(session) -> None:  # type: ignore[no-untyped-def]
    s, ref = session
    cal = load_calibration()
    r = check_track_width(s, ref, fog_sign=cal.fog_convention.angle_sign_to_ned_yaw)
    assert r.r2 > 0.9
    assert 0.5 < r.track_width_m < 0.7


def test_v5_integrity_all_sessions() -> None:
    dates = [d for d in load_split().dates() if _have(d)]
    if not dates:
        pytest.skip("NCLT data not downloaded")
    for d in dates:
        s = load_inputs(d)
        ref = load_reference_for(d)
        for rec in (s.imu, s.fog, s.wheels, s.gps):
            assert np.all(np.diff(rec.t_us) > 0), f"{d}: non-increasing timestamps"
        tg = ref.gt.t_us[ref.gt.valid]
        span = s.imu.t_us[-1] - s.imu.t_us[0]
        covered = min(tg[-1], s.imu.t_us[-1]) - max(tg[0], s.imu.t_us[0])
        assert covered / span > 0.95, f"{d}: reference covers too little of the session"
        assert ref.nodes.valid.sum() > 100
