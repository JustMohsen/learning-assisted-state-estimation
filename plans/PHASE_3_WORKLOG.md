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

## Steps

- [x] 1. Torch SO(3)/quaternion utilities + tests
- [x] 2. State, FilterConfig, HeadOutputs (neutral), initial covariance
- [x] 3. Propagation (mean + covariance) + tests (vs strapdown; Jacobians vs autograd)
- [x] 4. Wheel and GPS updates + tests (Jacobians vs autograd; linear case vs WLS)
- [x] 5. Runner (initialisation, exact-time event loop, NavEstimate output, diagnostics) + compute check
- [ ] 6. Simulation validation as tests (Monte Carlo covariance, clean accuracy, matched-noise NEES/NIS) + realistic-scenario run
- [ ] 7. Optional wheel-scale state (17th error coordinate) and minimum GPS update interval + tests
- [ ] 8. Real-data tooling: `scripts/run_baseline.py` (parallel sessions, injected/natural outages, GPS off), oracle residuals, NIS on real data
- [ ] 9. Tuning on training days (grid search, parallel)
- [ ] 10. Validation-day evaluation vs baselines; Phase 3 report, summary, review notes

## Questions for the user

- See [REVIEW_PHASES.md](./REVIEW_PHASES.md), section 4 (wheel scale, validation volume, headline metric) and the Phase 3 decisions. Results from steps 7–10 will inform them.

## Log

- 2026-11-15 ~02:05–02:45: steps 1–5 completed (see branch `phase3-prototype` history for details). Prototype simulation results: clean 2 cm / 0.1°; matched-noise NEES 2.4, NIS 1.0 / 2.0; nominal NEES 133 (correlated GPS); slip corrupts gyro-z bias; ~0.95 ms per IMU step.

## Summary

(written when Phase 3 completes)
