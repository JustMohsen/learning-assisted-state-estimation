# Review: Phases 1, 2 and 3

*Prepared 15 November 2026 (overnight, autonomous; Phase 3 finished at the user's later request). Everything below was decided by the agent and is open to change. Items marked **[needs you]** are decisions only you should make. Agreed review order: Phases 1–2 first (sections 2–4), then Phase 3 (section 5).*

## 1. Where things are

| What | Where |
|---|---|
| Phase 1 plan, log, report | [PHASE_1_DATA_FOUNDATION.md](./PHASE_1_DATA_FOUNDATION.md), [PHASE_1_WORKLOG.md](./PHASE_1_WORKLOG.md), [reports/phase1_audit/README.md](../reports/phase1_audit/README.md) |
| Phase 2 plan, log, report | [PHASE_2_SIM_EVAL.md](./PHASE_2_SIM_EVAL.md), [PHASE_2_WORKLOG.md](./PHASE_2_WORKLOG.md), [reports/phase2_eval/README.md](../reports/phase2_eval/README.md) |
| Phase 3 plan, log, report | [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md), [PHASE_3_WORKLOG.md](./PHASE_3_WORKLOG.md), [reports/phase3_baseline/README.md](../reports/phase3_baseline/README.md) |
| Code | `src/lase/data` (Phase 1), `src/lase/sim` and `src/lase/evaluation` (Phase 2), `src/lase/filter` (Phase 3) |
| Tests | 83 unit tests (CI) and 31 validation tests (`uv run pytest tests/validation`), all passing |

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

## 4a. Decisions from the review discussion

### Phase 1 (15 November 2026)

| # | Topic | User decision / outcome |
|---|---|---|
| 1 | Wheel speed scale | **Estimate it as a slow state** in the baseline (Phase 3). Initialise each session from the **final estimate of the previous session by date**; the first session (2012-01-08) starts at 1.0. To settle in Phase 3: initial uncertainty (larger for the first session, growing with time since the last run), a stored per-date chain usable by Phase 4, and diagnosis of why the prototype's wheel-scale state drifted to 1.08–1.14. |
| 2 | Validation volume | **Keep the agreed 22/2/3 split.** |
| 3 | Headline metric | Deferred to the Phase 3 discussion. |
| – | GPS gating | Robust gating agreed. |
| – | Reference trust | Clarified: graph nodes (~5 m apart) come from multi-session LiDAR scan matching + RTK + odometry (mostly independent of our sensors; ~8 cm σ). Dense poses between nodes are interpolated with NCLT's wheel+FOG odometry (our sensor family). Primary scoring stays at graph nodes. Proposed: add node-only drift (RPE) so short-window metrics are also LiDAR/RTK-anchored. |

## 5. Phase 3: analytical baseline (complete; review after Phases 1–2)

**What exists:**
- An error-state Kalman filter in PyTorch, verified against independent references:
  - the strapdown integrator (1e-9);
  - autograd Jacobians (1e-8 / 1e-10);
  - batch least squares (1e-10);
  - Monte Carlo covariance;
  - simulation consistency (NEES 2.4 with a matched noise model).
- Tuned on training days only (25 configurations, 156 runs) and scored on the validation days.
- Full report: [reports/phase3_baseline/README.md](../reports/phase3_baseline/README.md).

**Headline (validation days, all sensors):**
- position median 4.7–5.3 m, heading 1.2°, velocity 0.15–0.18 m/s;
- p95 8.5 m on one day, but **65 m on 2013-02-23** (a GPS re-acquisition failure, open problem 1);
- during injected outages, error grows about 5 m in 120 s, comparable to NCLT's own wheel+FOG odometry started from a perfect pose.

**Important things that changed while building it** (details in the report, section 4):
1. **Updates applied at their exact timestamps.** Without this, the 0.71 m lever arm on a pitching Segway caused 50° heading drift.
2. **Two FOG handling bugs found and fixed** (telescoping, extrapolation across 0.1 ms gaps).
3. **FOG noise taken from the datasheet.**
4. **Earth-rate compensation implemented but off:** the data appears already compensated.

**Decisions needed from you:**

| # | Question | Evidence | My recommendation |
|---|---|---|---|
| 7 | **[needs you] Soft non-holonomic constraint in baseline A.** The scope has the lateral no-slip constraint as stage 3 and starts the baseline without it. | Without it: p95 210–240 m, drift ~85 m per 60 s, one training session diverged to km. With it: p95 8.5 m (most days), drift 2–3 m per 60 s. | Include it in A (analytically, fixed sigma), and keep stage 3 as *learned trust* in it |
| 8 | **[needs you] Wheel scale** (question 1 above) | As an estimated state it drifted to 1.08–1.14 and doubled errors; fixed 1.054 works | Keep fixed for A; revisit as a learned/calibration family later |
| 9 | GPS re-acquisition after outages (open problem 1) | 2013-02-23: 64 m jump, then ~6 min rejecting good GPS | Require several mutually consistent fixes after an outage before accepting; consider robust GPS updates |
| 10 | Correlated GPS error / overconfidence (NEES ≫ 2) | 1 Hz + 5 m helps accuracy but not consistency | Leave to the learned GPS multipliers (Phase 5), or add a 2-state GPS error model to A. Discuss. |
| 11 | Unexplained FOG drift (+2–4° per 10 min; 27° on 2013-02-23) | Not Earth rate (wrong sign); wider bias prior hurt | Investigate together (FOG documentation, pitch geometry, reference) before Phase 4 |
| 12 | Compute for Phase 4 | ~2 ms per IMU step per process, ~3–4 min per session | Batch many chunks (B > 1) and/or `torch.compile`; measure in Phase 4 planning |

## 6. Suggested order for the discussion

1. Phases 1–2: questions 1–5 and the two reports' headline tables.
2. Phase 3: questions 7–11, using the Phase 3 report.
3. Then plan Phase 4 together (not started).
