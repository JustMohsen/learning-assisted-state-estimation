# Phase 2 Worklog

Execution log for [PHASE_2_SIM_EVAL.md](./PHASE_2_SIM_EVAL.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md).

## Decisions made

- **Simulator outputs real-data types:** SensorStreams + Reference (+ SimTruth), so the filter, training and evaluation run unchanged on simulated data.
- **Truth trajectory:** valid dense reference resampled to 100 Hz, zero-phase 4th-order Butterworth low-pass at 2 Hz, quintic splines on position and unwrapped roll/pitch/yaw; body rates from Euler rates (verified against C' = C[w]x). The spline is the truth. It matches the reference to ~1 mm median; Segway pitch reaches -17..+21 deg, confirming the need for 3D attitude.
- **Sensor timestamps:** simulated at the session's real timestamps (realistic jitter/gaps); natural GPS gaps kept by default (use_real_gps_gaps).
- **Error magnitudes:** defaults are plausible, not identified (Phase 1 measurements + datasheet-order noise): IMU acc/gyro noise 0.02 m/s^2 / 0.003 rad/s per sample, biases 0.05 m/s^2 / 0.003 rad/s + random walk; FOG angle noise 1e-5 rad, bias 5e-6 rad/s; wheels scale 1.054, track 0.560 m, noise 0.02 m/s; GPS white 1.5 m + Gauss-Markov 2.5 m (tau 60 s), 40% mode-2 time. Scenarios: clean, biased, nominal, slip, gps_degraded (configs/sim/).
- **Gravity:** WGS-84 normal gravity with free-air correction at the NCLT origin (9.8029 m/s^2), in lase.geometry.constants; shared by simulator and filter.
- **Record builders:** raw loaders refactored into make_imu/make_fog/make_wheels/make_gps (LOADER_VERSION 2) so simulated records get identical validity flags.
- **No scripts/simulate.py:** simulations take ~0.2 s per session, so they are generated in memory where needed instead of being stored.
- **Validation-only strapdown integrator (sim/strapdown.py, NumPy):** an independent second implementation used only to validate the simulator and, later, to cross-check the PyTorch filter's propagation. It is not part of the estimator (one-implementation principle preserved for the filter).
- **Estimate format and association:** NavEstimate (N/E position, N/E velocity, heading, optional 5x5 covariance at the body reference point). Association is causal: latest estimate at or before each reference time, max staleness 200 ms; unassociated times reduce 'coverage' rather than being scored.
- **Scoring sets:** graph nodes (primary) and a 10 Hz dense grid (sensitivity; also used for outage curves and RPE). Pooled metrics are sample-wise across sessions. Because GPS outliers dominate RMSE, medians and p95 are always reported alongside RMSE.
- **Injected outages:** 30/60/120 s windows, 3 per duration per session (seeded by date), only where valid GPS had no gap > 2 s inside the window and in the 180 s before it, separated by >= 180 s for recovery; GPS rows removed (not flagged). 232 windows over 27 sessions.
- **Outage reference points:** 'GPS hold' (repeat last fix: no dead reckoning) and 'odometry (oracle-anchored)' (NCLT odometry restarted from the true pose at each outage start; non-causal, diagnostic only).
- **Baselines:** raw GPS (antenna minus course-rotated lever arm; velocity/course from fixes >= 1 s apart) and NCLT odometry aligned by the agreed common initial-pose prior (SE(2) at the first reference time).
- **evo** added as a dev dependency for the E1 cross-check only.
- **Fitted error models (training days only; scripts/fit_sim_errors.py -> reports/phase2_eval/sim_fit.json):** GPS inliers = 1.44 m 'white' + 2.53 m Gauss-Markov (tau 55 s), very close to the earlier defaults. GPS errors > 10 m are episodic (about 22-27 runs/h, median 2.2 s, p90 47 s; only 32% isolated fixes). Gross episodes (> 50 m) about 3.4/h, median 14 s, offsets median 81 m, p90 1 km, max 5 km. Fix mode alternates almost per fix, with nearly equal accuracy (2.94 vs 2.78 m), so no separate mode-2 noise is needed. Held fixes 0.7-1%. Wheel mean-speed residual robust std 0.035 m/s; slip-like excursions are strongly session-dependent (session median 0.019/moving-min, pooled 0.09).
- **New simulator mechanisms:** episodic GPS errors (multipath: Poisson, log-normal durations, log-uniform 8-40 m offsets; gross: Poisson, log-normal durations, offset = 50 m + log-normal excess), optional fast Gauss-Markov for the 'white' GPS component (white_tau_s), and a separate yaw low-pass cutoff (lowpass_yaw_hz). All default to off, so existing scenarios are unchanged.
- **Truth-trajectory cutoffs for 'realistic' (trade-off, open question):** the reference contains spurious high-frequency roll/pitch/position content that, through the 0.71 m IMU lever arm, produces lateral and vertical accelerations several times larger than the real IMU sees. Position/roll/pitch 1 Hz + yaw 5 Hz best match the real accelerometer and yaw-rate content but remove real roll-rate dynamics (sim 0.017 vs real 0.10 rad/s). The likely cause is that smoothing position and attitude independently breaks the physical coupling (the Segway rolls about its wheel contact points). A physically coupled trajectory model would fix it.
- **IMU noise is not identified from data:** at rest the Segway balances (accelerometer x 0.065 m/s^2, gyro y/z 0.015-0.027 rad/s from motion); the quietest axes bound sensor noise (accelerometer z 0.002 m/s^2, gyro x 0.0025 rad/s). The realistic scenario keeps the nominal IMU noise (0.02 / 0.003).

## Steps

- [x] 1. Truth trajectory: resample, zero-phase low-pass, splines; exact derivatives and body rates
- [x] 2. Sensor models: IMU (lever arm), FOG (released convention), wheels (scale, track, slip), GPS (Gauss–Markov, bursts, outliers, outages, held)
- [x] 3. Scenarios (`configs/sim/`), `SimTruth`, outputs as `SensorStreams` + `Reference`; `scripts/simulate.py`
- [x] 4. Simulator validation S1–S3 (+ tests)
- [x] 5. Evaluation core: `NavEstimate`, causal association, accuracy metrics (nodes + dense)
- [x] 6. Outage injection, outage error-growth curves, relative pose error
- [x] 7. Consistency statistics (NEES/NIS, coverage), plots
- [x] 8. Baselines (raw GPS, NCLT odometry) + `evo` cross-check (E1–E2)
- [x] 9. Phase 2 report + summary; update README status

### Hardening (added 15 November ~02:55, after the user paused Phase 3)

- [x] 10. Fit simulator error models to real data (training days only): GPS residual decomposition (white / Gauss–Markov / tau, outliers and gross-outlier episodes, fix modes, held fixes), wheel residuals and slip-like excursions, at-rest IMU variability, FOG residuals → `configs/sim/realistic.yaml` + report
- [x] 11. Simulator validation across several sessions and scenarios (S1–S3 + realism comparison: simulated vs real residual statistics) + tests
- [ ] 12. Standard evaluation entry point (`evaluate_estimate`: nodes/dense metrics, RPE, injected and natural outage curves, consistency) + tests
- [ ] 13. Harness edge cases (empty or short estimates, NaN fields, truncated sessions, no outage candidates) + tests
- [ ] 14. Update the Phase 2 report; write `plans/REVIEW_PHASES.md` (Phases 1–2 decisions, questions, Phase 3 draft decisions)

## Questions for the user

- **Simulator error magnitudes** are plausible defaults, not identified from data. If you want sim scenarios tied to measured values, Phase 3's oracle-residual analysis can supply them (e.g. GPS error statistics, IMU noise at rest); I'd update configs/sim then.
- **Reporting with outliers:** I report median/p95 alongside RMSE because raw-GPS RMSE is dominated by km-scale outliers. Do you want a headline metric fixed now (e.g. RMSE at graph nodes as in the scope, plus median), or keep reporting all?

## Log

- 2026-11-15 01:50: Phase 2 started: plan written (plans/PHASE_2_SIM_EVAL.md); run control in plans/AUTONOMOUS_RUN.md; wake-ups every 20 min + one-shot 04:30.
- 2026-11-15 01:55: Step 1 done: sim/trajectory.py (truth splines, exact derivatives, body rates, FOG body-z angle antiderivative).
- 2026-11-15 01:55: Step 2 done: sim/sensors.py (IMU with lever-arm acceleration, FOG released convention, wheels with scale/track/slip, GPS with Gauss-Markov, bursts, outliers, outages, held, mode Markov chain).
- 2026-11-15 01:55: Step 3 done: sim/config.py + 5 scenarios, sim/simulate.py (SimRun = SensorStreams + Reference from truth + SimTruth); 2013-01-10 simulates in ~0.2 s per scenario; 8 synthetic-circle unit tests (exact lever-arm specific force, wheel kinematics, FOG convention, outages, random-walk statistics).
- 2026-11-15 01:57: Step 4 done (2012-01-22, nominal): S1 clean strapdown error 8.4 mm per 10 s at 47 Hz, 0.53 mm at 4x rate, convergence ratio 16.06 (second order). S2 Phase 1 cross-checks recover injected values: GPS median 3.41 m (injected 3.41), wheel scale 1.0546 (1.054), raw track 0.5905 m (0.590), FOG/IMU scale -0.996/+0.996 with sub-ms lag. S3 noise std 0.0201/0.00299 vs 0.02/0.003. tests/validation/test_sim.py (3 pass).
- 2026-11-15 01:57: Finding: heading-increment scale is ~0.996 even in simulation with exact sensors, because Euler yaw rate differs from body z rate when the Segway pitches (up to ~20 deg); the real-data 0.998 is therefore consistent with unit scale.
- 2026-11-15 02:03: Step 5 done: evaluation/estimate.py (NavEstimate, causal association, reference samples at nodes/dense, reference-as-estimate) and evaluation/metrics.py (ErrorSeries, pooled summaries, planar RPE).
- 2026-11-15 02:03: Step 6 done: evaluation/outages.py (plan_outages, withhold_gps, outage_curves).
- 2026-11-15 02:03: Step 7 done: evaluation/consistency.py (NEES/NIS, chi-square summaries) and evaluation/plots.py (trajectories, error vs time with outage shading, outage growth).
- 2026-11-15 02:03: Step 8 done: baselines.py + scripts/evaluate_baselines.py -> reports/phase2_eval/. E1: position RMSE equals evo APE exactly. E2: reference scores itself at 6 mm (nodes, causal association) / 0 (dense). Pooled (test days): raw GPS median 4.0 m but RMSE 336 m (outliers); NCLT odometry RMSE 75 m, heading 15 deg. Outages (232): GPS hold median 15/62/109 m at 10/60/120 s; oracle-anchored odometry 0.22/1.0/2.0 m. 7 unit + 2 validation tests.
- 2026-11-15 02:03: Finding: consumer GPS has extreme outliers flagged as valid fixes: 1.4% of samples > 100 m and 0.4% > 1 km (up to 5.9 km), in 15 of 27 sessions. Robust GPS gating is mandatory in Phase 3.
- 2026-11-15 02:03: Step 9 done: reports/phase2_eval/README.md; README status updated; Phase 2 summary written.
- 2026-11-15 02:37: User instruction (~02:50): do not implement Phase 3; harden Phase 2; plan Phase 3 for discussion. Phase 3 prototype moved to branch phase3-prototype and reverted on main; hardening steps 10-14 added.
- 2026-11-15 02:37: Step 10 done: sim/fit.py, scripts/fit_sim_errors.py, configs/sim/realistic.yaml (+ simulator GPS episodes, fast GM, yaw cutoff).
- 2026-11-15 02:37: Step 11 done: scripts/validate_sim_realism.py (real vs simulated twins, 6 training sessions; reports/phase2_eval/sim_realism_*.csv). Realistic vs real (session medians): GPS median 0.995x, inlier sigma 1.06x, time > 10 m 0.94x, gross episodes 0.90x, held 0.97x, fix-mode fraction 1.06x, wheel residual 1.10x, wheel excursions 1.04x, yaw-rate content 1.00x, forward acceleration 0.87x; mismatches: > 10 m runs 1.7x, lateral/vertical acceleration 1.8x/2.4x, roll rate 0.18x (trajectory trade-off). Nominal scenario for comparison: no episodes or slip, 4.5x/5.9x accelerations. Simulator validation S1-S3 now runs on 3 sessions (train/train/validation, different seasons) plus a realistic-scenario GPS statistics test: 29 validation tests pass.

## Summary

**Phase 2 is complete** (15 November 2026, overnight). All 9 steps are done.

**Built:**
- `lase.sim`: truth trajectory, sensor models, scenarios, `simulate_session`, a validation-only strapdown integrator and checks S1–S3.
- `lase.evaluation`: estimate format, causal association, metrics and RPE, outage planning and curves, NEES/NIS, plots, baselines.
- Scripts: `scripts/evaluate_baselines.py`.

**Tests:** 60 unit tests (CI) and 22 validation tests (Phase 1 + Phase 2), all passing.

**Key results** ([reports/phase2_eval/README.md](../reports/phase2_eval/README.md)):
- The simulator validates: second-order strapdown convergence, injected values recovered by the Phase 1 checks, and correct noise statistics.
- The harness matches `evo` exactly.
- Baseline reference points:
  - raw GPS: median 3.7–4.5 m, but RMSE in the hundreds of metres because of extreme outliers;
  - NCLT odometry: 54–111 m RMSE without GPS;
  - oracle-anchored wheels + FOG during injected outages: 1.0 m (60 s) and 2.0 m (120 s) median.

**Most consequential decisions:**
- the simulator emits real-data types;
- simulated sensors use real timestamps;
- error magnitudes are plausible defaults, not identified values;
- association is causal with 200 ms staleness;
- outage windows are placed only where GPS was healthy;
- oracle-anchored odometry is labelled diagnostic-only.

**Needs the user:** see *Questions for the user* (none blocking).
