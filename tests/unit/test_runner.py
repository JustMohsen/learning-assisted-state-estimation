"""End-to-end filter test on a synthetic circle with clean simulated sensors (no data needed)."""

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from lase.data.calibration import load_calibration
from lase.data.conversions import unwrap_fog
from lase.data.geodesy import LocalProjection
from lase.data.raw import make_fog, make_gps, make_imu, make_wheels
from lase.data.session import SensorStreams
from lase.evaluation.estimate import reference_samples
from lase.evaluation.metrics import compute_errors, summarize
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.runner import initial_condition_from_reference, run_filter
from lase.geometry.constants import earth_rate_ned
from lase.sim.config import FogErrors, GpsErrors, WheelErrors
from lase.sim.sensors import simulate_fog, simulate_gps, simulate_imu, simulate_wheels
from lase.sim.simulate import _reference_from_truth
from lase.sim.trajectory import truth_from_reference
from tests.unit.test_sim_sensors import T0, _circle_reference, _zero_imu


def _clean_streams(wheel_scale: float = 1.0):  # type: ignore[no-untyped-def]
    cal = load_calibration()
    traj = truth_from_reference(_circle_reference(120.0))
    rng = np.random.default_rng(0)
    jitter = lambda n, step: (  # noqa: E731
        T0 + 15_000_000 + np.cumsum(rng.integers(int(step * 0.8), int(step * 1.2), n))
    ).astype(np.int64)
    t_imu, t_fog, t_wh, t_gps = (
        jitter(4000, 21_000),
        jitter(8500, 10_500),
        jitter(3800, 23_000),
        jitter(560, 143_000),
    )
    w_ie = np.asarray(earth_rate_ned())  # sensors sense the Earth rate; the filter removes it
    imu = simulate_imu(traj, t_imu, cal, _zero_imu(), rng, w_ie)
    fog = simulate_fog(
        traj,
        t_fog,
        FogErrors(angle_noise_rad=0, bias_init_radps=0, bias_rw_radps_sqrts=0),
        rng,
        w_ie,
    )
    wh = simulate_wheels(traj, t_wh, WheelErrors(speed_scale=wheel_scale, noise_mps=0), rng)
    gps = simulate_gps(
        traj, t_gps, cal, GpsErrors(white_sigma_m=0, gm_sigma_m=0, mode2_frac=0), rng
    )
    n = len(gps.t_us)
    nan = np.full(n, np.nan)
    streams = SensorStreams(
        date=dt.date(2000, 1, 1),
        imu=make_imu(t_imu, np.zeros((len(t_imu), 3)), imu.acc, imu.gyr),
        fog=make_fog(t_fog, fog.angle_released),
        wheels=make_wheels(t_wh, wh.left, wh.right),
        gps=make_gps(
            gps.t_us, gps.mode, np.zeros(n, dtype=np.int64), gps.lat, gps.lon, gps.alt, nan, nan
        ),
        fog_angle=unwrap_fog(fog.angle_released),
        gnss_ned=LocalProjection.from_config(cal.local_frame).to_local(gps.lat, gps.lon, gps.alt),
    )
    return streams, _reference_from_truth(traj), cal


@pytest.fixture(scope="module")
def clean_circle():  # type: ignore[no-untyped-def]
    return _clean_streams(1.0)


@pytest.mark.parametrize("use_fog", [True, False])
def test_clean_circle_tracks_truth(clean_circle, use_fog: bool) -> None:  # type: ignore[no-untyped-def]
    streams, ref, cal = clean_circle
    cfg = FilterConfig(use_fog=use_fog)
    params = replace(FilterParams.build(cal, cfg), wheel_scale=1.0)
    ic = initial_condition_from_reference(ref, T0 + 20_000_000)
    res = run_filter(streams, ic, cfg, params, t_end_us=T0 + 95_000_000)
    s = summarize(
        compute_errors(
            res.estimate, reference_samples(ref, "dense", t_window=(ic.t_us, T0 + 95_000_000))
        )
    )
    assert s["coverage"] > 0.99
    assert s["pos_rmse_m"] < 0.05
    # Causal association compares each reference time with an estimate up to one IMU
    # interval (~21 ms) older; at 0.1 rad/s that alone is up to ~0.12 deg of heading.
    assert s["heading_rmse_deg"] < 0.2
    assert s["vel_rmse_mps"] < 0.02
    assert res.wheel["accepted"].all()
    assert res.gnss["accepted"].all()


def test_wheel_scale_state_converges() -> None:
    streams, ref, cal = _clean_streams(1.08)
    base = FilterConfig()
    cfg = base.model_copy(update={"prior": base.prior.model_copy(update={"wheel_scale": 0.05})})
    params = replace(FilterParams.build(cal, cfg), wheel_scale=1.054)  # Phase 1 estimate as start
    ic = initial_condition_from_reference(ref, T0 + 20_000_000)
    res = run_filter(streams, ic, cfg, params, t_end_us=T0 + 95_000_000)
    assert res.biases["wheel_scale"][-1] == pytest.approx(1.08, abs=0.003)
    fixed = run_filter(streams, ic, base, params, t_end_us=T0 + 95_000_000)
    assert np.allclose(fixed.biases["wheel_scale"], 1.054)  # zero prior: stays fixed


def _gross_after_outage(streams: SensorStreams, offset_m: float) -> SensorStreams:
    """GPS outage 40-60 s, then a consistent offset for 15 s, then good fixes."""
    import dataclasses

    from lase.evaluation.outages import withhold_gps

    s = withhold_gps(streams, [(T0 + 40_000_000, T0 + 60_000_000)])
    ned = s.gnss_ned.copy()
    bad = (s.gps.t_us >= T0 + 60_000_000) & (s.gps.t_us < T0 + 75_000_000)
    ned[bad, 0] += offset_m
    return dataclasses.replace(s, gnss_ned=ned)


def _worst_error(clean_circle, reacq: bool, offset_m: float) -> tuple[float, int]:  # type: ignore[no-untyped-def]
    streams, ref, cal = clean_circle
    base = FilterConfig()
    gnss = base.gnss.model_copy(update={"reacq_enabled": reacq, "sigma_m": 5.0})
    cfg = base.model_copy(update={"gnss": gnss})
    params = replace(FilterParams.build(cal, cfg), wheel_scale=1.0)
    ic = initial_condition_from_reference(ref, T0 + 20_000_000)
    res = run_filter(
        _gross_after_outage(streams, offset_m), ic, cfg, params, t_end_us=T0 + 95_000_000
    )
    e = compute_errors(
        res.estimate, reference_samples(ref, "dense", t_window=(ic.t_us, T0 + 95_000_000))
    )
    return float(np.nanmax(e.pos_err_norm)), res.gnss_reacquisitions


def test_gps_reacquisition_rejects_gross_fixes_after_outage(clean_circle) -> None:  # type: ignore[no-untyped-def]
    worst_rule, n_reacq = _worst_error(clean_circle, True, 18.0)
    worst_plain, _ = _worst_error(clean_circle, False, 18.0)
    assert worst_rule < 2.0  # the consistent-but-implausible 18 m offset is never accepted
    assert n_reacq >= 1  # good GPS is re-acquired after the bad episode
    assert worst_plain > 3 * worst_rule  # the plain gate lets the offset pull the estimate
