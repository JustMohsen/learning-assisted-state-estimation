import numpy as np
import pytest

from lase.data.calibration import load_calibration
from lase.data.reference import GraphNodes, GroundTruth, Reference
from lase.geometry.constants import GRAVITY_MPS2
from lase.sim.config import FogErrors, GpsErrors, ImuErrors, SlipConfig, WheelErrors
from lase.sim.sensors import (
    in_spans,
    poisson_spans,
    random_walk,
    simulate_fog,
    simulate_gps,
    simulate_imu,
    simulate_wheels,
)
from lase.sim.trajectory import truth_from_reference

V, R = 1.0, 10.0  # speed (m/s), radius (m): clockwise circle seen from above (right turn)
W = V / R
T0 = 1_000_000_000_000_000


def _circle_reference(duration_s: float = 120.0) -> Reference:
    t = T0 + np.arange(0, int(duration_s * 1e6), 10_000, dtype=np.int64)
    s = (t - T0) * 1e-6
    psi = W * s
    p = np.stack([R * np.sin(psi), R * (1 - np.cos(psi)), np.zeros_like(s)], axis=1)
    rpy = np.stack([np.zeros_like(s), np.zeros_like(s), np.angle(np.exp(1j * psi))], axis=1)
    gt = GroundTruth(t_us=t, p_w=p, rpy=rpy, flags=np.zeros(len(t), dtype=np.uint16))
    nodes = GraphNodes(
        t_us=t[:1],
        cov=np.eye(6)[None],
        gt_index=np.zeros(1, dtype=np.int64),
        gt_offset_us=np.zeros(1, dtype=np.int64),
        valid=np.ones(1, dtype=bool),
    )
    return Reference(gt=gt, nodes=nodes)


@pytest.fixture(scope="module")
def traj():  # type: ignore[no-untyped-def]
    return truth_from_reference(_circle_reference())


def _mid(n: int = 2000) -> np.ndarray:
    return T0 + np.linspace(30e6, 90e6, n).astype(np.int64)  # avoid filter edge effects


def _zero_imu() -> ImuErrors:
    return ImuErrors(
        acc_noise_mps2=0,
        gyr_noise_radps=0,
        acc_bias_init_mps2=0,
        gyr_bias_init_radps=0,
        acc_bias_rw_mps2_sqrts=0,
        gyr_bias_rw_radps_sqrts=0,
    )


def test_truth_kinematics(traj) -> None:  # type: ignore[no-untyped-def]
    st = traj.at(_mid())
    np.testing.assert_allclose(np.linalg.norm(st.v_w[:, :2], axis=1), V, atol=1e-4)
    np.testing.assert_allclose(st.omega_b[:, 2], W, atol=1e-5)


def test_imu_clean_circle_with_lever_arm(traj) -> None:  # type: ignore[no-untyped-def]
    cal = load_calibration()
    imu = simulate_imu(traj, _mid(), cal, _zero_imu(), np.random.default_rng(0))
    lx, ly, _ = cal.sensors.imu.l_b
    expected = np.array([-(W**2) * lx, V * W - W**2 * ly, -GRAVITY_MPS2])
    np.testing.assert_allclose(imu.acc, np.tile(expected, (len(imu.acc), 1)), atol=2e-4)
    np.testing.assert_allclose(imu.gyr[:, 2], W, atol=1e-5)


def test_wheels_scale_and_track(traj) -> None:  # type: ignore[no-untyped-def]
    err = WheelErrors(speed_scale=1.05, track_width_m=0.56, noise_mps=0)
    wh = simulate_wheels(traj, _mid(), err, np.random.default_rng(0))
    np.testing.assert_allclose(wh.left, 1.05 * (V + W * 0.28), atol=1e-4)
    np.testing.assert_allclose(wh.right, 1.05 * (V - W * 0.28), atol=1e-4)
    assert not wh.slip.any()


def test_wheel_slip_marks_episodes(traj) -> None:  # type: ignore[no-untyped-def]
    err = WheelErrors(noise_mps=0, slip=SlipConfig(rate_per_min=10.0, asymmetric_prob=0.0))
    wh = simulate_wheels(traj, _mid(), err, np.random.default_rng(1))
    assert wh.slip.any()
    clean = 1.054 * V
    mean_speed = 0.5 * (wh.left + wh.right)
    assert np.all(mean_speed[wh.slip] > clean * 1.05)


def test_fog_released_convention(traj) -> None:  # type: ignore[no-untyped-def]
    t = _mid()
    fog = simulate_fog(
        traj,
        t,
        FogErrors(angle_noise_rad=0, bias_init_radps=0, bias_rw_radps_sqrts=0),
        np.random.default_rng(0),
    )
    assert np.all((fog.angle_released >= 0) & (fog.angle_released < 2 * np.pi))
    rate = -np.diff(np.unwrap(fog.angle_released)) / (np.diff(t) * 1e-6)  # undo the -1 sign
    np.testing.assert_allclose(rate, W, atol=1e-4)


def test_gps_outages_removed_and_clean_position(traj) -> None:  # type: ignore[no-untyped-def]
    cal = load_calibration()
    t = T0 + np.arange(30_000_000, 90_000_000, 140_000, dtype=np.int64)
    err = GpsErrors(
        white_sigma_m=0,
        gm_sigma_m=0,
        mode2_frac=0,
        outage_rate_per_hour=120,
        outage_duration_s=(5, 10),
    )
    g = simulate_gps(traj, t, cal, err, np.random.default_rng(2))
    assert len(g.t_us) < len(t)
    assert not in_spans(g.t_us, g.outage_spans).any()
    np.testing.assert_allclose(g.error_ned, 0, atol=1e-9)


def test_random_walk_statistics() -> None:
    t = np.arange(0, 100_000_001, 10_000, dtype=np.int64)  # 100 s
    rng = np.random.default_rng(0)
    finals = np.array([random_walk(t, 0.0, 0.01, 1, rng)[-1, 0] for _ in range(400)])
    assert finals.std() == pytest.approx(0.01 * np.sqrt(100), rel=0.15)


def test_poisson_spans_non_overlapping() -> None:
    spans = poisson_spans(0, 3_600_000_000, 1 / 60, (5, 20), np.random.default_rng(0))
    assert 30 < len(spans) < 90
    assert all(b > a for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
