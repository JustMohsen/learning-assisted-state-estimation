# Phase 1 Report: Data Audit and Cross-Sensor Validation

*Generated 15 November 2026 from all 27 NCLT sessions. Plan: [PHASE_1_DATA_FOUNDATION.md](../../plans/PHASE_1_DATA_FOUNDATION.md). Execution log and decisions: [PHASE_1_WORKLOG.md](../../plans/PHASE_1_WORKLOG.md).*

Reproduce:

```bash
uv run python scripts/download_nclt.py --split all   # ~5.4 GB download, ~8 min
uv run python scripts/audit_nclt.py                  # sessions.csv, spans.json, figures
uv run python scripts/validate_phase1.py             # validation.csv, figures
uv run pytest tests/validation                       # pass/fail checks V1-V5
```

## Files

| File | Content |
|---|---|
| `sessions.csv` | Per-session audit: durations, rates and interval statistics, validity, GPS quality and outages, stationary periods, reference coverage, graph nodes |
| `spans.json` | Per-session time spans (µs): GPS outages over 5 s, stationary periods, input gaps |
| `validation.csv` | Per-session results of checks V1–V4 |
| `durations_outages.png` | Session lengths by split, with natural GPS outage time |
| `sample_intervals.png` | Sample-interval distributions per sensor, all sessions pooled |
| `v1_gps_overlay.png` | GPS fixes over the reference trajectory (two sessions) |
| `v2_v3_scatter.png` | Heading changes (FOG, IMU) and wheel speed against the reference |

## Headline results

**The data is clean and consistent, and every frame, sign, timing and scale check passes.** Values marked *estimated* are now recorded in [configs/data/nclt_calibration.yaml](../../configs/data/nclt_calibration.yaml).

| Check | Result (27 sessions) | Status |
|---|---|---|
| V1 GPS vs reference | Median horizontal error 2.9–5.6 m; 95th percentile 8–94 m (multipath outliers); inlier fit rotation ≤ 0.32°, scale error ≤ 0.53% | Pass. The frame definition and lever arm are right. |
| V2 heading timing, sign and scale | IMU gyro z: scale +0.988…+0.9995, lag +0.6…+2.5 ms. **FOG: scale −0.988…−1.000**, lag −0.9…−2.0 ms | Pass. **The FOG angle has the opposite sign to NED yaw.** |
| V3 wheel speed | Wheel speed / reference forward speed = **1.044–1.085**, increasing over the year | Pass, with a finding: wheels read ~5% high, and the scale drifts |
| V4 track width | Raw-units width 0.578–0.611 m (R² 0.96–0.99); **physical width 0.560 m** (std 3 mm) after dividing by the wheel scale | Pass. Value now estimated. |
| V5 integrity | No backwards or duplicate timestamps in any stream; reference covers more than 95% of each session | Pass |

## Data facts

| Topic | Finding |
|---|---|
| Volume | 36.3 h in total: train 30.8 h (22 days), validation 1.6 h (2 days), test 3.9 h (3 days) |
| Short sessions | **2013-01-10 (validation) is 17 min.** 2012-04-29 (43 min) and 2012-06-15 (55 min) are **truncated releases**: their covariance files list 365 and 334 graph nodes beyond the released ground truth. These are now masked as invalid. |
| Logged rates | IMU ~47 Hz, FOG ~94.5 Hz, wheels 37–56 Hz (varies by session), GPS 4.4–9 Hz. Wheels have 2–23 gaps over 0.2 s per session (up to 0.78 s). |
| GPS | Fix mode is 2 (horizontal only) for ~20–53% of fixes and 3 otherwise; 2012-04-29 is almost all mode 3. **Satellite count is always 0.** Held positions 0–2%. Invalid rows ≈ 0, except 2012-02-18 (1.6%). |
| Natural GPS outages | Over 5 s: 4–18 min per session (1–6 outages), longest 837 s (2012-02-18). 2012-04-29 has none. A natural outage test set is available. |
| Reference | ~148 Hz (not 100 Hz). Starts 0–1 s after the sensors, ends up to 65 s before them. No internal gaps over 1 s. A few NaN rows (flagged). |
| Graph nodes | Every 3.6 s / ~5 m (not the 8 m the reader comments mention); marginal position std ~8 cm; nearest dense pose within 20 ms |
| Stationary periods | The Segway self-balances: wheels jitter up to ~0.08 m/s at rest. With a smoothed-speed heuristic, 3–17 stationary periods (1–11 min) per session. |
| Reverse motion | 1.5–4% of wheel samples are clearly reversing |

## Answers to the plan's open questions

1. **Archive contents:** identical in every session. Twelve files: `gps`, `gps_rtk`, `gps_rtk_err`, `ms25`, `ms25_euler`, `kvh`, `wheels`, `odometry_mu`, `odometry_mu_100hz`, `odometry_cov`, `odometry_cov_100hz`, plus a README.
2. **Reference-frame origin:** fixed for the whole dataset at 42.293227°, −83.709657°, 270 m, with the paper's own linearisation (Sec. 3, Eqs. 1–3, Table 3). The official GPS reader does *not* use it: it re-centres on each file's first fix for plotting.
3. **Mounting geometry:** from paper Table 4. IMU and FOG axes are aligned with the body (forward-right-down). Lever arms from the body origin (axle centre): IMU (−0.11, −0.18, −0.71) m, FOG (0, −0.25, −0.49) m, GPS antenna (0, −0.25, −0.51) m.
4. **Units and signs:** accelerometer m/s² with z down (≈ −9.8 at rest), gyros rad/s with standard signs, **FOG reversed (×−1)**, wheels m/s with negative values in reverse.
5. **GPS fix modes:** 0 = not yet seen, 1 = no fix, 2 = latitude/longitude good, 3 = altitude good too (paper Table 7). Held fixes are rare (≤ 2%) and stay informational, since stationary periods also produce them.
6. **Reference coverage:** essentially the whole session. Small trimming at the start (≤ 1 s) and end (≤ 65 s), apart from the truncated sessions noted above.
7. **The short 2013-01-10 validation day:** the split stays as agreed. Validation therefore contains 97 minutes, about 80 of them from 2013-02-23.

## Implications for later phases

- **The GPS update needs robust gating.** Multipath outliers reach tens of metres, and 5–15% of fixes are more than 10 m off. Mode-2 fixes may deserve separate noise.
- **Wheel scale is not 1 and drifts over months** (1.044 → 1.085). This directly supports learning or estimating wheel scale (scope stage 5). A fixed value of 1 would bias dead reckoning by about 5%.
- **FOG and IMU timing match the reference to about 2 ms.** No time-offset estimation is needed initially.
- **The FOG sign must be applied** (`fog_convention.angle_sign_to_ned_yaw = -1`). The event-stream payload still carries the released angle; the filter applies the sign.
- **The reference is not independent of the FOG and wheels.** It is odometry-interpolated between nodes, which is consistent with the near-zero FOG lag. Prefer graph-node scoring, as planned.
