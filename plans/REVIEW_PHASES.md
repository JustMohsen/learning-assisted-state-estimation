# Review: Phases 1 and 2, and the Phase 3 Draft Plan

*Prepared 15 November 2026 (overnight, autonomous) for the morning discussion. Everything below was decided by the agent and is open to change. Items marked **[needs you]** are decisions only you should make.*

## 1. Where things are

| What | Where |
|---|---|
| Phase 1 plan, log, report | [PHASE_1_DATA_FOUNDATION.md](./PHASE_1_DATA_FOUNDATION.md), [PHASE_1_WORKLOG.md](./PHASE_1_WORKLOG.md), [reports/phase1_audit/README.md](../reports/phase1_audit/README.md) |
| Phase 2 plan, log, report | [PHASE_2_SIM_EVAL.md](./PHASE_2_SIM_EVAL.md), [PHASE_2_WORKLOG.md](./PHASE_2_WORKLOG.md), [reports/phase2_eval/README.md](../reports/phase2_eval/README.md) |
| Phase 3 **draft** plan (not started) | [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md) (section 7: prototype lessons and decisions to make) |
| Phase 3 prototype (paused, not merged) | Git branch `phase3-prototype` |
| Code | `src/lase/data` (Phase 1), `src/lase/sim` and `src/lase/evaluation` (Phase 2) |
| Tests | 63 unit tests (CI) and 29 validation tests (`uv run pytest tests/validation`), all passing |

## 2. Phase 1: data foundation (complete)

**Built:**
- All 27 NCLT sessions downloaded and checksummed into `local_data/` (36.3 h; 5.5 GB raw + 6.9 GB extracted).
- Loaders that flag bad rows instead of dropping them.
- A sourced calibration config.
- GPS → local frame conversion.
- The reference trajectory (dense poses + graph nodes).
- Role-separated data access and a causal event stream.
- An all-session audit and cross-sensor validation (V1–V5, all pass).

**Key decisions:**
1. **Frame origin:** the dataset paper's fixed origin and linearised conversion (Sec. 3, Table 3), which *define* the reference frame. (The official GPS reader re-centres per file; my first assumption, that it held the anchor, was wrong.)
2. **Mounting:** from paper Table 4. IMU and FOG are aligned with the body; lever arms are published.
3. **FOG sign:** NED yaw = −1 × released FOG angle. Measured in all 27 sessions (scale −0.988…−1.000).
4. **Wheel geometry:** physical track width 0.560 m (estimated). The wheel speed scale is 1.054 (training median) and **drifts from 1.044 to 1.085 over the year**.
5. **Truncated releases** (2012-04-29, 2012-06-15) stay in training; nodes past the released ground truth are masked.
6. **Parquet cache deferred:** a session parses in ~0.1 s.

**Findings that matter later:**
- **GPS has extreme outliers flagged as valid:** 1.4% of samples are over 100 m off, 0.4% over 1 km. This makes robust gating mandatory.
- **IMU and FOG timing agree with the reference to ~2 ms.**
- **The Segway self-balances:** wheels jitter at rest, and pitch reaches ±20°.
- **The reference is odometry-interpolated** between graph nodes about 5 m / 3.6 s apart.

## 3. Phase 2: simulator and evaluation harness (complete + hardened)

**Simulator:**
- Real reference trajectories → smooth truth → sensors at the real timestamps, with known injected errors.
- Same output types as real data.
- Six scenarios: `clean`, `biased`, `nominal`, `slip`, `gps_degraded` and **`realistic`** (fitted to the training days).
- Validated:
  - second-order strapdown convergence;
  - the Phase 1 checks recover the injected values;
  - noise statistics match;
  - checked on 3 sessions.

**Harness:**
- `NavEstimate` with causal association.
- Graph-node and dense metrics, RPE.
- Injected and natural GPS-outage curves, NEES/NIS.
- Plots and a single `evaluate_estimate` entry point.
- Matches `evo` exactly.
- Baselines: raw GPS, NCLT odometry, GPS-hold and oracle-anchored odometry.

**Key decisions:**
1. Simulated sensors use the real timestamps, and real GPS gaps are kept.
2. Causal association: the latest estimate at or before each reference time (≤ 200 ms old).
3. Injected outages of 30/60/120 s, placed only where GPS was healthy, with seeded positions.
4. Medians and p95 are reported alongside RMSE, because GPS outliers dominate RMSE.
5. The `realistic` scenario's truth-trajectory smoothing is a **trade-off**: position/roll/pitch at 1 Hz, yaw at 5 Hz. It matches the real accelerometer and yaw content but loses real roll dynamics.

**Reference points for Phase 3** (all sessions):

| Situation | Number |
|---|---|
| Raw GPS, median / p95 | ~3.7–4.5 m / 15–43 m |
| NCLT odometry without GPS (whole session), RMSE | 54–111 m |
| Injected outages, oracle-anchored wheels + FOG, median | 1.0 m at 60 s, 2.0 m at 120 s |
| Natural outages (84; median 122 s, max 837 s), oracle-anchored odometry, median | 2.6 m at 120 s, 4.6 m at 300 s |
| Holding the last GPS fix at the start of a natural outage | already 46 m off: last fixes before going indoors are corrupted |

## 4. Decisions needed from you

| # | Question | My recommendation |
|---|---|---|
| 1 | **[needs you] Wheel speed scale:** the scope fixes it in the pilot, but it is 1.044–1.085 and drifts | Estimate it as a slow state in the analytical baseline (cheap, physical, fair to all methods); keep the learned wheel residual for fast effects |
| 2 | **[needs you] Validation volume:** 97 min, ~80 of them from one day | Keep the agreed split; if checkpoint selection proves noisy, add cross-validation over training days rather than changing the split |
| 3 | **[needs you] Headline metric** | Keep the scope's RMSE at graph nodes, and always show median and p95 next to it (outliers dominate RMSE) |
| 4 | **Simulator trajectory trade-off** (realistic: 1 Hz / 5 Hz cutoffs) | Accept for now. If simulation must reproduce roll dynamics, build a physically coupled Segway trajectory model (rolling about the contact points). |
| 5 | **Outage test design:** injected 30/60/120 s vs natural ~2–14 min | Report both. Consider adding a 300 s injected duration to match the natural ones. |
| ~~6~~ | ~~Separate noise for GPS mode-2 fixes~~ | **Answered by data:** accuracy is nearly equal (2.94 vs 2.78 m) and modes alternate per fix, so no separate noise is needed |

## 5. Phase 3: to plan together

The draft plan ([PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md)) proposes an error-state Kalman filter in PyTorch, as agreed in the scope.

Before your instruction to pause, I had built a **prototype** (branch `phase3-prototype`). It is reverted on `main`; nothing depends on it. What it established, as input to the plan:
- **Wheel/GPS updates must be applied at their exact timestamps.** Applying them to a ~20 ms-old state, combined with the 0.71 m lever arm on a pitching Segway, caused 50° of heading drift in 5 minutes on perfect data.
- **FOG increments must telescope exactly,** or the error leaks into the bias estimates.
- **With a correct noise model the filter is consistent:** NEES 2.4 (ideal 2) and NIS 1.0 / 2.0 (ideal 1 / 2).
- **With the real, time-correlated GPS error it becomes overconfident:** NEES 133.
- **Slip corrupts the gyro bias** if not handled.
- **Speed:** ~1 ms per IMU step, about 4 min per session.

Decisions to make together:
1. the wheel scale (question 1 above);
2. how to handle correlated GPS error: an error state, inflation/decimation, or the learned multipliers;
3. slip robustness in the baseline;
4. whether to adopt the exact-time update scheme;
5. initialisation;
6. the tuning method.

## 6. Suggested order for the discussion

1. Questions 1–3 above.
2. Skim the two reports' headline tables.
3. Phase 3 decisions 2–4, informed by the prototype.
4. Whether to start Phase 3 from the prototype branch or from scratch.
