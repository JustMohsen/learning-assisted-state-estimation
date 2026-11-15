from pathlib import Path

import numpy as np

from lase.data.reference import load_reference


def _write_session(tmp_path: Path) -> tuple[Path, Path]:
    # 10 s straight line heading East (yaw = +pi/2) at 2 m/s, 100 Hz, first row NaN.
    t = np.arange(0, 10_000_001, 10_000, dtype=np.int64) + 1_000_000_000_000_000
    s = (t - t[0]) * 1e-6
    rows = [f"{t[0]},nan,nan,nan,nan,nan,nan"]
    rows += [
        f"{ti}, 0.0, {2.0 * si:.6f}, 0.0, 0.0, 0.0, {np.pi / 2:.9f}"
        for ti, si in zip(t[1:], s[1:], strict=True)
    ]
    gt = tmp_path / "gt.csv"
    gt.write_text("\n".join(rows) + "\n")
    cov_vals = ",".join(["0.01"] + ["0.0"] * 20)
    cov = tmp_path / "cov.csv"
    cov.write_text(
        f"{t[100] + 3},{cov_vals}\n{t[500] - 2},{cov_vals}\n{t[-1] + 5_000_000},{cov_vals}\n"
    )
    return gt, cov


def test_reference_nodes_and_velocity(tmp_path: Path) -> None:
    gt_path, cov_path = _write_session(tmp_path)
    ref = load_reference(gt_path, cov_path)
    assert not ref.gt.valid[0]
    assert ref.nodes.gt_index.tolist()[:2] == [100, 500]
    assert ref.nodes.gt_offset_us.tolist()[:2] == [-3, 2]
    assert ref.nodes.valid.tolist() == [True, True, False]  # node beyond the GT span
    assert ref.nodes.cov.shape == (3, 6, 6)
    np.testing.assert_allclose(ref.nodes.cov[:, 0, 0], 0.01)
    tq = ref.gt.t_us[200:800:50]
    v, mask = ref.velocity_w(tq)
    assert mask.all()
    np.testing.assert_allclose(v[:, :2], np.tile([0.0, 2.0], (len(tq), 1)), atol=1e-6)
    fwd, m2 = ref.forward_speed(tq)
    assert m2.all()
    np.testing.assert_allclose(fwd, 2.0, atol=1e-6)
    _, mask_out = ref.velocity_w(np.array([ref.gt.t_us[-1] + 1_000_000]))
    assert not mask_out[0]
