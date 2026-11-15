"""Fit the wheel speed scale as a linear trend in date (training days only).

    uv run python scripts/fit_wheel_scale_trend.py

Uses the per-session scale measured against the reference in Phase 1 (check V3,
reports/phase1_audit/validation.csv; training split only) and prints the values for
``filter.wheel.scale_trend_*`` in configs/parameters.yaml. Validation days are shown for
information only (they are not used in the fit).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from lase.data.splits import Split, load_split
from lase.utils.paths import repo_root

REF_DATE = dt.date(2012, 1, 1)


def main() -> int:
    v3 = pd.read_csv(repo_root() / "reports" / "phase1_audit" / "validation.csv")
    v3["date"] = pd.to_datetime(v3["date"]).dt.date
    split = load_split()
    train = v3[v3["date"].isin(split.dates(Split.TRAIN))]
    years = np.array([(d - REF_DATE).days / 365.25 for d in train["date"]])
    slope, intercept = np.polyfit(years, train["v3_scale"].to_numpy(), 1)
    resid = train["v3_scale"].to_numpy() - (intercept + slope * years)
    print(f"reference date {REF_DATE}, {len(train)} training sessions")
    print(f"scale_trend_intercept: {intercept:.5f}")
    print(f"scale_trend_per_year:  {slope:.5f}")
    print(f"residual std {resid.std(ddof=2):.4f}, max |residual| {np.abs(resid).max():.4f}")
    for d in split.dates(Split.VALIDATION):
        row = v3[v3["date"] == d]
        pred = intercept + slope * (d - REF_DATE).days / 365.25
        meas = float(row["v3_scale"].iloc[0]) if len(row) else float("nan")
        print(f"validation {d}: trend {pred:.4f}, measured {meas:.4f} (information only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
