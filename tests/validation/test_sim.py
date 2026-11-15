"""Simulator validation S1-S3 on a real session's trajectory (needs downloaded data)."""

import datetime as dt

import numpy as np
import pytest

from lase.data.calibration import load_calibration
from lase.data.crosscheck import check_gps, check_track_width, check_wheel_speed, check_yaw_rate
from lase.data.session import extracted_dir
from lase.sim.config import load_scenario
from lase.sim.simulate import SimRun, simulate_session
from lase.sim.validation import check_imu_noise, check_strapdown

pytestmark = pytest.mark.validation
# Sessions from different seasons and splits (train, train, validation).
DATES = [dt.date(2012, 1, 22), dt.date(2012, 8, 4), dt.date(2013, 2, 23)]


@pytest.fixture(scope="module", params=DATES, ids=str)
def run(request: pytest.FixtureRequest) -> SimRun:
    d = request.param
    if not (extracted_dir(d) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")
    return simulate_session(d, load_scenario("nominal"))


def test_s1_strapdown_convergence(run: SimRun) -> None:
    r = check_strapdown(run.truth.trajectory, load_calibration())
    assert r.native_err_m < 0.05  # cm-level drift over 10 s from discretisation only
    assert 10 < r.convergence_ratio < 25  # second-order scheme: ~16 for 4x rate


def test_s2_crosschecks_recover_injected_values(run: SimRun) -> None:
    s, ref, truth = run.streams, run.reference, run.truth
    cal = load_calibration()
    v1 = check_gps(s, ref, cal)
    injected = float(np.median(np.linalg.norm(truth.gps_error_ned[:, :2], axis=1)))
    assert v1.median_err_m == pytest.approx(injected, rel=0.02)
    assert abs(v1.rotation_deg) < 0.2
    v2 = check_yaw_rate(s, ref)
    assert v2.fog_scale == pytest.approx(truth.config.fog.released_sign, abs=0.01)
    assert v2.imu_scale == pytest.approx(1.0, abs=0.01)
    assert abs(v2.fog_lag_ms) < 2
    v3 = check_wheel_speed(s, ref)
    assert v3.scale == pytest.approx(truth.wheel_speed_scale, abs=0.005)
    v4 = check_track_width(s, ref, fog_sign=truth.config.fog.released_sign)
    assert v4.track_width_m == pytest.approx(
        truth.wheel_speed_scale * truth.track_width_m, rel=0.01
    )


def test_s3_noise_statistics(run: SimRun) -> None:
    err = load_scenario("nominal").imu
    r = check_imu_noise(run.truth.trajectory, load_calibration(), run.streams.imu.t_us[:20000], err)
    assert r.acc_noise_std == pytest.approx(r.acc_noise_cfg, rel=0.03)
    assert r.gyr_noise_std == pytest.approx(r.gyr_noise_cfg, rel=0.03)


def test_realistic_scenario_reproduces_fitted_gps_statistics() -> None:
    """The fitted 'realistic' scenario matches the real GPS error level and episode rate."""
    from lase.data.session import load_inputs, load_reference_for
    from lase.sim.fit import session_stats

    d = dt.date(2012, 5, 26)
    if not (extracted_dir(d) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")
    cal = load_calibration()
    cfg = load_scenario("realistic")
    sim = simulate_session(d, cfg)
    s = session_stats(sim.streams, sim.reference, cal, cfg.wheels.speed_scale)
    real = session_stats(load_inputs(d), load_reference_for(d), cal, cal.wheels.speed_scale or 1.0)
    assert s["gps_median_err_m"] == pytest.approx(real["gps_median_err_m"], rel=0.25)
    assert s["gps_inlier_sigma_m"] == pytest.approx(real["gps_inlier_sigma_m"], rel=0.25)
    assert 0.0 < s["gps_frac_gt10m"] < 0.3
    assert s["gps_gross_per_h"] > 0
