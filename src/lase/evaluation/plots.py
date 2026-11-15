"""Standard evaluation figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lase.evaluation.estimate import NavEstimate, ReferenceSamples
from lase.evaluation.metrics import ErrorSeries
from lase.evaluation.outages import OutageCurves, Span


def plot_trajectories(
    ref: ReferenceSamples, ests: Sequence[NavEstimate], path: Path, title: str = ""
) -> None:
    """Overlay of estimated and reference horizontal trajectories."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ref.pos_ne[:, 1], ref.pos_ne[:, 0], "k-", lw=1.0, label="reference")
    for e in ests:
        ax.plot(e.pos_ne[:, 1], e.pos_ne[:, 0], lw=0.8, label=e.name)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors_vs_time(
    series: dict[str, ErrorSeries], path: Path, spans: Sequence[Span] = (), title: str = ""
) -> None:
    """Horizontal position error vs time, with (injected) outage spans shaded."""
    fig, ax = plt.subplots(figsize=(11, 4))
    t0 = min(int(s.t_us[0]) for s in series.values())
    for a, b in spans:
        ax.axvspan((a - t0) * 1e-6 / 60, (b - t0) * 1e-6 / 60, color="#d97706", alpha=0.2)
    for name, s in series.items():
        ax.plot((s.t_us - t0) * 1e-6 / 60, s.pos_err_norm, lw=0.8, label=name)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("horizontal error (m)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_outage_growth(curves: dict[str, OutageCurves], path: Path, title: str = "") -> None:
    """Median (line) and 90th-percentile (dashed) error vs time since outage start."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        if len(c.errors_m) == 0:
            continue
        (line,) = ax.plot(c.offsets_s, c.percentile(50), marker="o", label=f"{name} (median)")
        ax.plot(c.offsets_s, c.percentile(90), "--", color=line.get_color(), label=f"{name} (p90)")
    ax.set_xlabel("time since GPS outage start (s)")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _ = np
