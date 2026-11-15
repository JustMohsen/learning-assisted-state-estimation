# Phase 3 Worklog

Execution log for [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md). **Phase 4 must not be started before user review.**

History:
- **~02:00:** steps 1–5 were built during the autonomous run.
- **~02:50:** the user paused Phase 3. The code moved to branch `phase3-prototype` and was reverted on `main`.
- **Morning:** the user asked to resume and finish Phase 3, then review it after Phases 1–2. The prototype code was restored onto `main` from the branch. It is compatible with the hardened Phase 2, and all 78 unit tests pass.

## Decisions made

- **Filter numerics:** PyTorch float64 on CPU, batch-first. SO(3) and quaternion operations use Taylor branches at small angles, so values and gradients stay finite.
- **IMU/FOG fusion:** body-z rotation = inverse-variance combination of the sign-corrected FOG increment and IMU gyro z (FOG weight ~0.99); IMU-only if no FOG data.
- **Exact-time updates:** measurements are buffered to the next IMU sample. The interval is split at measurement times, with interpolated IMU and FOG values. Adds ≤ 1 IMU interval of latency; uses no future data. (A stale-state version caused 50° heading drift on clean data via the lever arm.)
- **FOG increments telescope exactly:** the boundary angle is reused.
- **Exact first-order transition:** includes the bias effect through the midpoint attitude and uses the exact SO(3) right Jacobian (F = autograd to 1e-8).
- **Disclosed approximations:**
  - FOG angle noise is modelled as white rate noise;
  - the wheel update uses the corrected body rate as an exogenous input.
- **Updates:** gated (99.9%) Joseph form with attitude reset. GPS recovery: inflate the position covariance after 30 s of rejections.
- **Initial condition:** reference pose at first valid reference time + 1 s (common prior); wheel-based velocity; zero biases.
- **Resuming without the user's input on the open decisions (reversible choices, alternatives measured):**
  - wheel scale as an optional estimated state (off by default);
  - correlated GPS handled by tuning sigma and the minimum GPS update interval;
  - tuning by grid search on training days only;
  - validation days for scoring only;
  - test days untouched.
- **Wheel scale as error-state index 16 (CONVENTIONS updated, N_ERR = 17):** h = s * v_fwd; with zero prior and zero random walk it is exactly fixed (default; calibration value 1.054), with a prior it is estimated. This keeps the user's decision open and lets both be measured.
- **Minimum GPS update interval (gnss.min_interval_s, default 0):** one lever against time-correlated GPS error, tuned in step 9 together with the GPS sigma.
- **Bug fix, FOG extrapolation:** real FOG samples sometimes arrive ~0.1 ms apart, so a two-sample extrapolation rate produced one-sample heading spikes of up to 180 deg (cancelled next step by telescoping but corrupting the bias estimates: gyro-z bias up to 0.05 rad/s). The rate is now taken over a >= 20 ms baseline (unit test added).
- **FOG noise from the KVH DSP-3000 datasheet:** angle random walk 2e-5 rad/sqrt(s) (0.0667 deg/sqrt(h)), bias prior 1e-5 rad/s (bias stability ~1 deg/h). The earlier defaults were ~10x too loose, which let GPS errors drag the heading.
- **Optional soft non-holonomic constraint (nonholonomic.enabled; scope question for the user):** pseudo-measurement that axle lateral and vertical body velocities are ~0 (sigma 0.1 m/s), applied at each wheel sample. The agreed scope lists the (learned-trust) lateral no-slip constraint as stage 3 and starts the baseline without it. Without it, lateral velocity is unobserved except via GPS: on real data velocity error 0.6-0.9 m/s, one session diverged to km, and 30-min GPS-free runs reach 49-446 m. With it: velocity 0.25-0.36 m/s, no divergence, GPS-free 31-59 m (comparable to NCLT odometry). Both variants are kept and reported.
- **Tuning protocol:** training days only (6 sessions spread over the year, 30-min segments starting 5 min in, GPS withheld in the injected outage windows). Ranked by the normalised sum of node position p95, heading RMSE and median error at outage end; ties broken by NIS consistency. Validation days are not used for selection.
- **Tuned values (stages 1-2, 22 configurations, 138 runs):** GPS sigma 5 m with 1 Hz GPS updates (handles correlated GPS error better than sigma 3 m or 7 Hz), wheel sigma 0.15 m/s, accelerometer noise density 0.05 m/s^2/sqrt(Hz), non-holonomic constraint on. The optimum is flat: wheel 0.3 / acc 0.05-0.2 score within ~1%, but wheel sigma 0.3 gives wheel NIS 0.03-0.14 (too pessimistic) vs 0.55. Training-segment medians: position 3.1 m median, 5.7 m p95, heading 1.36 deg, velocity 0.18 m/s, outage-end error 3.2 m.
- **Wheel-scale state stays off (finding for the user's decision):** estimating it (prior 0.02, rw 1e-5) drifted to 1.08-1.14, beyond the 1.044-1.076 the reference shows for these days, and doubled the errors (p95 13.7 vs 5.7 m), so it absorbs other model errors and is not cleanly identifiable in this baseline.
- **GPS recovery after 300 s (stage 3):** recovery after 90 s caused both large failures (2012-02-04 p95 95 m, max 113 m; 2012-11-17 max 22 m). With 300 s (or never) no recovery triggers on the tuning sessions and those drop to p95 10 m / max 6.6 m. 300 s keeps a safety net for genuine divergence.
- **Earth rotation compensated (modelling fix found on the validation days):** in every session the FOG-integrated heading drifted +2 to +4 deg per 10 min against the reference. The KVH FOG senses the Earth rate (Omega sin(42.3 deg) = 4.9e-5 rad/s = 1.7 deg per 10 min on the vertical), five times the datasheet FOG-bias prior, so it could not be absorbed and dragged the heading whenever GPS was weak. The filter now subtracts C_wb^T w_ie from gyro and FOG (with its attitude Jacobian; F still equals autograd to 1e-8). The simulator senses the Earth rate too (SimConfig.earth_rotation, default on), and the validation strapdown has the matching option. Transport rate and Coriolis acceleration (< 1e-4 m/s^2 at Segway speeds) are neglected.
- **Earth-rate compensation: implemented but OFF by default (reverses the previous entry).** On six training days (not the tuning ones) compensating made heading worse in every session (no GPS: 8.3 vs 6.4 deg RMSE over 20 min; with GPS 1.45 vs 1.28 deg), and on the validation days too. The observed FOG drift against the reference is positive (+2 to +4 deg per 10 min), opposite in sign to an uncompensated Earth rate (-1.7 deg per 10 min), so the released NCLT FOG data appears to be Earth-rate compensated already. The capability stays (FilterConfig.earth_rotation, SimConfig.earth_rotation), default off in both. Open question for the user: confirm from the NCLT/KVH documentation.
- **FOG-bias prior stays at the datasheet value (1e-5 rad/s):** wider priors (5e-5, 1.5e-4) were worse on training days (heading 1.6/1.8 vs 1.37 deg; outage-end error 3.4/3.8 vs 2.0 m).
- **Parameters file (R0, user request):** configs/parameters.yaml holds every tunable value of the estimator, data handling and evaluation, each commented (what, units, source: tuned/datasheet/measured/design), loaded by lase.parameters.load_parameters with experiment overrides from configs/experiments/ (deep merge; unknown keys rejected). Former hard-coded values moved in: GPS recovery rejection count, FOG extrapolation limit/baseline, start-up offset, evaluation staleness, dense rate, RPE intervals, outage planning and curve settings, reference node tolerance and velocity window. configs/filter/baseline.yaml removed (its values are in parameters.yaml). tests/unit/test_parameters.py enforces completeness and comments; CONVENTIONS rule added. Analysis-only thresholds in one-off Phase 1/2 diagnostics remain documented keyword arguments.
- **Non-holonomic constraint in baseline A, lateral only (user decision; R1):** lateral = along the axle, pitch-independent. The vertical term (body-z velocity ~ 0) is wrong under pitch (v sin(pitch), ~0.07 m/s typical) and is now optional, off by default. New learned output nhc_var_mult (CONVENTIONS 6.3): B learns the constraint's noise as a constant, C learns its trust; lateral skids are handled like forward slip (scope stage 3 rewritten accordingly).
- **Wheel model (R2, user's encoder model):** reported speed = s * V_roll + R_w * q, with q = body pitch rate (gyro y) and R_w = wheel.encoder_pitch_coef_m (0.24 m default: data 0.17-0.24 m, about the wheel radius; to be retuned). V_roll = f . v_axle with f = normalize(axle_w x down): the horizontal direction perpendicular to the axle, from attitude. A first version using the signed norm of the axle velocity in the body x-z plane was rejected: it made the wheel prediction depend on the poorly observed vertical velocity (no barometer, horizontal GPS) and biased the wheel-scale estimate (1.074 vs 1.08 true in a test). Analytic Jacobian checked against autograd (1e-10).

## Steps

- [x] 1. Torch SO(3)/quaternion utilities + tests
- [x] 2. State, FilterConfig, HeadOutputs (neutral), initial covariance
- [x] 3. Propagation (mean + covariance) + tests (vs strapdown; Jacobians vs autograd)
- [x] 4. Wheel and GPS updates + tests (Jacobians vs autograd; linear case vs WLS)
- [x] 5. Runner (initialisation, exact-time event loop, NavEstimate output, diagnostics) + compute check
- [x] 6. Simulation validation as tests (Monte Carlo covariance, clean accuracy, matched-noise NEES/NIS) + realistic-scenario run
- [x] 7. Optional wheel-scale state (17th error coordinate) and minimum GPS update interval + tests
- [x] 8. Real-data tooling: `scripts/run_baseline.py` (parallel sessions, injected/natural outages, GPS off), oracle residuals, NIS on real data
- [x] 9. Tuning on training days (grid search, parallel)
- [x] 10. Validation-day evaluation vs baselines; Phase 3 report, summary, review notes

### Revision after the user's review (15 November 2026)

- [x] R0. Single commented parameters file (`configs/parameters.yaml`) for every tunable value; move existing magic numbers into it; CONVENTIONS rule "every decided value is a named, commented parameter"
- [x] R1. Non-holonomic constraint in A, **lateral only**; scope/plan updated: B learns its noise, C learns its trust, lateral slip handled like forward slip
- [x] R2. Wheel model: encoder senses wheel rotation relative to the chassis, `reported speed = s * V_axle,perp + R_w * q_gyro_y` (R_w a parameter, ~0.2 m measured); pitch-independent axle speed; vertical constraint dropped or loosened
- [ ] R3. GPS re-acquisition: require N mutually consistent fixes after an outage (parameters)
- [ ] R4. GPS error states (2-D Gauss-Markov, fitted values) for correlated GPS error
- [ ] R5. FOG investigation (warm-up, season, pitch/roll correlation, KVH documentation), then a fitted FOG bias model
- [ ] R6. Wheel-scale test: fixed 1.054 vs chained slow state (previous session's final estimate; first session 1.0), on the corrected model
- [ ] R7. Retune on training days; re-evaluate on validation days; update report and review notes

## Questions for the user

- See [REVIEW_PHASES.md](./REVIEW_PHASES.md), section 4 (wheel scale, validation volume, headline metric) and the Phase 3 decisions. Results from steps 7–10 will inform them.

## Log

- 2026-11-15 ~02:05–02:45: steps 1–5 completed (see branch `phase3-prototype` history for details). Prototype simulation results: clean 2 cm / 0.1°; matched-noise NEES 2.4, NIS 1.0 / 2.0; nominal NEES 133 (correlated GPS); slip corrupts gyro-z bias; ~0.95 ms per IMU step.
- 2026-11-15 02:55: Step 6 done: Monte Carlo covariance unit test (400 runs, variance ratios within 0.72-1.3; 2 s), validation tests for clean accuracy and matched-noise consistency (pass). Untuned baseline on the fitted 'realistic' scenario: median 2.9 m, p95 6.2 m, max 31.6 m (GPS episodes partly accepted), GPS NIS 8.4, NEES 126; gyro x/y biases recovered (8e-5 vs 3e-3 rad/s true RMS).
- 2026-11-15 02:57: Step 7 done: wheel-scale state + minimum GPS update interval; test: with a prior the state converges from 1.054 to the true 1.08 (+-0.003) on the clean circle; with zero prior it stays fixed. 80 unit tests pass.
- 2026-11-15 03:19: Step 8 (real-data tooling) first runs (30-min segments, 4 training days): untuned baseline 7-11 deg heading error and 0.6-0.9 m/s velocity error with GPS; 2012-02-18 diverged (km); without GPS hundreds of metres in 30 min. Diagnosed below.
- 2026-11-15 03:19: Findings: IMU x/y axes validated against the reference (correlations 0.67-0.999 on the diagonal, no swaps or sign flips). No consistent IMU-body yaw offset (median heading error within +-1 deg, mixed sign). Remaining heading error of ~1.3-2.5 deg RMSE comes from (a) time-correlated GPS errors accumulating into heading (1 Hz GPS updates reduce it), and (b) attitude limits: the FOG alone, integrating body-z rotation as yaw, has 1.3 deg RMSE over 10 min because the Segway pitches up to 20 deg. The GPS recovery rule (30 s of rejections) can accept gross-error fixes, since gross episodes last up to ~60 s.
- 2026-11-15 03:19: Step 8 done: filter/experiment.py (run_session / run_many: parallel processes, modes gps/outages/nogps, segment windows, evaluate_estimate + NIS/acceptance rows).
- 2026-11-15 04:29: Step 9 in progress: stage 1 (16 configs x 6 sessions, 47 min) and stage 2 (7 configs) done -> reports/phase3_baseline/tuning_*.csv. 2012-02-04 failure traced: a 4-minute gross GPS episode (offsets 300 -> 50 m); the filter rejected it for ~100 s, then the 90 s recovery rule inflated the covariance and accepted a fix 112 m off. Stage 3 tests recovery after 90 s / 300 s / never.
- 2026-11-15 04:38: Step 9 done: tuned configuration written to configs/filter/baseline.yaml (GPS 5 m at 1 Hz, recovery 300 s, wheel 0.15 m/s, acc 0.05, non-holonomic on). 3 stages, 25 configurations, 156 training-segment runs.
- 2026-11-15 04:53: First validation-day evaluation (before the fix below): 2013-01-10 median 4.7 m / p95 8.5 m / heading 1.2 deg; 2013-02-23 median 5.3 m but p95 65 m, and without GPS heading 27.5 deg (NCLT's own odometry also 27 deg there). Non-holonomic constraint essential (without it median 7-11 m, p95 210-240 m).
- 2026-11-15 05:34: Remaining FOG drift of ~+3 deg per 10 min (~9e-5 rad/s) looks like a genuine FOG bias, ~10x the datasheet-based prior (1e-5 rad/s), so the filter cannot learn it even with GPS. Testing wider FOG-bias priors (1e-5 / 5e-5 / 1.5e-4) on training days.
- 2026-11-15 06:00: Step 10 done: validation-day evaluation (scripts/evaluate_baseline.py -> reports/phase3_baseline/validation_*), tuned config on all sim scenarios (scripts/sim_baseline.py), Phase 3 report, REVIEW_PHASES section 5, README status. 2013-02-23 p95 65 m traced to GPS re-acquisition after a natural outage: grown uncertainty let ~67 m-wrong fixes through, then good fixes were rejected until the 300 s recovery.
- 2026-11-15 15:04: R0 done: configs/parameters.yaml, lase/parameters.py, configs/experiments/ (README, no_nhc.yaml), wiring through runner, experiment, evaluation and data loading; 88 unit tests pass.
- 2026-11-15 15:06: R1 done: lateral-only constraint (vertical optional), nhc_var_mult in HeadOutputs, scope / development plan / conventions / review notes updated; 89 unit tests pass.
- 2026-11-15 15:11: R2 done: pitch-independent rolling speed, encoder pitch-rate term, vertical constraint off by default (R1); 90 unit tests pass.

## Summary

**Phase 3 is complete** (15 November 2026, ~06:30), pending review.

**Built:** `lase.filter` with the following modules:
- `so3`, `state`, `params`, `learned`, `propagation`, `updates` (wheel, GPS, non-holonomic);
- `runner` (exact-time event loop);
- `validation`, `experiment`.

Scripts: `tune_baseline.py`, `evaluate_baseline.py`, `sim_baseline.py`. Tuned configuration: now in `configs/parameters.yaml` (R0; was `configs/filter/baseline.yaml`).

**Tests:** 83 unit + 31 validation, all passing.

**Results:** see [reports/phase3_baseline/README.md](../reports/phase3_baseline/README.md).
- Validation days: median 4.7–5.3 m, heading 1.2°.
- One re-acquisition failure (p95 65 m on 2013-02-23).
- Outage drift growth comparable to NCLT odometry.

**Needs the user:** [REVIEW_PHASES.md](./REVIEW_PHASES.md) section 5 (questions 7–12):
- the non-holonomic constraint in A (scope);
- wheel scale;
- GPS re-acquisition;
- correlated GPS;
- the FOG drift;
- compute.

Phase 4 not started.
