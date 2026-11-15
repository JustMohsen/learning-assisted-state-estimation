"""Run the Phase 1 all-session audit and write reports/phase1_audit/.

uv run python scripts/audit_nclt.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lase.data.audit import audit_session
from lase.data.session import load_inputs, load_reference_for
from lase.data.splits import load_split
from lase.utils.paths import repo_root


def main() -> int:
    out = repo_root() / "reports" / "phase1_audit"
    out.mkdir(parents=True, exist_ok=True)
    split = load_split()
    rows, all_spans = [], {}
    dts: dict[str, list[np.ndarray]] = {"imu": [], "fog": [], "wheel": [], "gps": []}
    for d in split.dates():
        s = load_inputs(d)
        ref = load_reference_for(d)
        row, spans = audit_session(s, ref, split.split_of(d).value)
        rows.append(row)
        all_spans[d.isoformat()] = spans
        for k, rec in [("imu", s.imu), ("fog", s.fog), ("wheel", s.wheels), ("gps", s.gps)]:
            dts[k].append(np.diff(rec.t_us) * 1e-3)
        print(
            f"{d}: {row['duration_min']:.1f} min, GPS outage {row['gps_outage_total_min']:.1f} min",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "sessions.csv", index=False, float_format="%.4g")
    (out / "spans.json").write_text(json.dumps(all_spans, indent=1))

    # Figure 1: duration and GPS outage time per session
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(df))
    colors = df["split"].map({"train": "#3b6fd8", "validation": "#d97706", "test": "#16a34a"})
    ax.bar(x, df["duration_min"], color=colors, label="duration")
    ax.bar(x, df["gps_outage_total_min"], color="#111827", alpha=0.6, label="GPS outage (>5 s)")
    ax.set_xticks(x, df["date"], rotation=70, fontsize=8)
    ax.set_ylabel("minutes")
    ax.set_title("Session duration (blue train, orange validation, green test) and GPS outage time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "durations_outages.png", dpi=120)
    plt.close(fig)

    # Figure 2: sample-interval distributions, all sessions pooled
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (k, v) in zip(axes, dts.items(), strict=True):
        d = np.concatenate(v)
        d = d[(d > 0) & (d < np.percentile(d, 99.9))]
        ax.hist(d, bins=200, color="#3b6fd8")
        ax.set_yscale("log")
        ax.set_title(f"{k} sample interval (ms)")
    fig.tight_layout()
    fig.savefig(out / "sample_intervals.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
