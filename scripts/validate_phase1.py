"""Run Phase 1 cross-sensor validation checks (V1-V4) on all sessions.

    uv run python scripts/validate_phase1.py

Writes reports/phase1_audit/validation.csv and figures. Track-width and FOG-sign
estimates from *training* sessions are printed for the calibration config.
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lase.data.calibration import load_calibration
from lase.data.crosscheck import (
    check_gps,
    check_track_width,
    check_wheel_speed,
    check_yaw_rate,
    common_grid,
    heading_increments,
)
from lase.data.session import load_inputs, load_reference_for
from lase.data.splits import Split, load_split
from lase.geometry.euler import euler_zyx_to_matrix
from lase.utils.paths import repo_root

FOG_SIGN = -1.0  # established by V2; written to configs/data/nclt_calibration.yaml


def _plot_examples(out_dir, dates) -> None:  # type: ignore[no-untyped-def]
    cal = load_calibration()
    fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 5.5))
    for ax, d in zip(np.atleast_1d(axes), dates, strict=True):
        s, ref = load_inputs(d), load_reference_for(d)
        m = ref.gt.valid
        ax.plot(ref.gt.p_w[m, 1], ref.gt.p_w[m, 0], color="#3b6fd8", lw=0.8, label="reference")
        g = s.gps.valid
        ax.scatter(s.gnss_ned[g, 1], s.gnss_ned[g, 0], s=0.5, color="#d97706", label="GPS")
        ax.set_aspect("equal")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title(f"V1 {d}")
        ax.legend(markerscale=10)
    fig.tight_layout()
    fig.savefig(out_dir / "v1_gps_overlay.png", dpi=120)
    plt.close(fig)

    d = dates[-1]
    s, ref = load_inputs(d), load_reference_for(d)
    grid = common_grid(s, ref, 10.0)
    d_gt, d_fog, d_imu = heading_increments(s, ref, grid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(d_gt, FOG_SIGN * d_fog, s=0.3, alpha=0.3)
    axes[0].set_title(f"V2 {d}: FOG (sign-corrected), 1-s heading change")
    axes[1].scatter(d_gt, d_imu, s=0.3, alpha=0.3)
    axes[1].set_title("V2: IMU gyro z vs reference")
    w = s.wheels
    v_ref, mask = ref.forward_speed(grid, half_window_us=500_000)
    v_w = np.interp(grid, w.t_us, 0.5 * (w.left + w.right))
    axes[2].scatter(v_ref[mask], v_w[mask], s=0.3, alpha=0.3)
    axes[2].plot([-1, 3], [-1, 3], "k--", lw=0.8)
    axes[2].set_title("V3: wheel speed vs reference forward speed (m/s)")
    for ax in axes[:2]:
        ax.plot([-0.7, 0.7], [-0.7, 0.7], "k--", lw=0.8)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_xlabel("reference heading change (rad)")
    fig.tight_layout()
    fig.savefig(out_dir / "v2_v3_scatter.png", dpi=120)
    plt.close(fig)
    _ = cal, euler_zyx_to_matrix


def main() -> int:
    out = repo_root() / "reports" / "phase1_audit"
    out.mkdir(parents=True, exist_ok=True)
    cal = load_calibration()
    split = load_split()
    rows = []
    for d in split.dates():
        s, ref = load_inputs(d), load_reference_for(d)
        row: dict[str, object] = {"date": d.isoformat(), "split": split.split_of(d).value}
        row.update({f"v1_{k}": v for k, v in asdict(check_gps(s, ref, cal)).items()})
        row.update({f"v2_{k}": v for k, v in asdict(check_yaw_rate(s, ref)).items()})
        row.update({f"v3_{k}": v for k, v in asdict(check_wheel_speed(s, ref)).items()})
        row.update(
            {f"v4_{k}": v for k, v in asdict(check_track_width(s, ref, fog_sign=FOG_SIGN)).items()}
        )
        rows.append(row)
        print(
            f"{d}: GPS med {row['v1_median_err_m']:.2f} m | FOG lag {row['v2_fog_lag_ms']:+.1f} ms "
            f"scale {row['v2_fog_scale']:+.4f} | IMU lag {row['v2_imu_lag_ms']:+.1f} ms "
            f"scale {row['v2_imu_scale']:+.4f} | wheel scale {row['v3_scale']:.3f} | "
            f"track {row['v4_track_width_m']:.3f} m (R2 {row['v4_r2']:.3f})",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df["v1_median_residual_n_m"] = [r["v1_median_residual_ne_m"][0] for r in rows]  # type: ignore[index]
    df["v1_median_residual_e_m"] = [r["v1_median_residual_ne_m"][1] for r in rows]  # type: ignore[index]
    df = df.drop(columns=["v1_median_residual_ne_m"])
    df.to_csv(out / "validation.csv", index=False, float_format="%.5g")
    train = df[df["split"] == Split.TRAIN.value]
    print(
        f"\nTraining-day estimates: track width median {train['v4_track_width_m'].median():.4f} m "
        f"(IQR {train['v4_track_width_m'].quantile(0.25):.4f}-{train['v4_track_width_m'].quantile(0.75):.4f}); "
        f"FOG scale median {train['v2_fog_scale'].median():+.4f}; "
        f"wheel scale median {train['v3_scale'].median():.4f}"
    )
    _plot_examples(out, [split.dates(Split.VALIDATION)[0], split.dates(Split.TRAIN)[2]])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
