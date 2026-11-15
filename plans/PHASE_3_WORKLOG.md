# Phase 3 Worklog

Execution log for [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md). **Phase 4 must not be started before user review.**

## Decisions made

- **Filter numerics:** PyTorch float64 on CPU, batch-first; SO(3)/quaternion operations with Taylor branches at small angles so values and gradients stay finite (needed for Phase 4 training).
- **Default noise densities (to be tuned):** accelerometer 0.02 m/s^2/sqrt(Hz) (includes Segway vibration), gyro 0.002 rad/s/sqrt(Hz), FOG 2e-4 rad/sqrt(s), bias random walks 1e-4 / 1e-5 / 1e-7. Wheel sigma 0.1 m/s; GPS sigma 3 m; gates at 99.9%. Initial prior: position 0.5 m, velocity 0.3 m/s, attitude 0.02 rad, biases 0.1 m/s^2 / 0.01 rad/s / 1e-4 rad/s.
- **HeadOutputs interface implemented now with neutral defaults** (zero corrections, unit multipliers), so the baseline and learned methods share one code path from the start.

## Steps

- [x] 1. Torch SO(3)/quaternion utilities + tests
- [x] 2. State, FilterConfig, HeadOutputs (neutral), initial covariance
- [ ] 3. Propagation (mean + covariance) + tests (vs strapdown; Jacobians vs autograd)
- [ ] 4. Wheel and GPS updates + tests (Jacobians vs autograd; linear case vs WLS)
- [ ] 5. Runner (initialisation, event loop, NavEstimate output, diagnostics) + compute check
- [ ] 6. Simulation validation (clean, nominal, slip, gps_degraded): tracking, bias convergence, NEES
- [ ] 7. Real-data validation: oracle residuals, NIS, baselines comparison, injected outages
- [ ] 8. Tuning on training days; validation-day scores; Phase 3 report and summary

## Questions for the user

## Log

- 2026-11-15 02:05: Phase 3 started after Phase 2 completed: plan written (plans/PHASE_3_BASELINE.md).
- 2026-11-15 02:07: Step 1 done: filter/so3.py (skew, Exp/Log, quaternion product/exp/normalise, matrix<->quaternion, yaw) with tests against SciPy and gradient finiteness at zero.
- 2026-11-15 02:07: Step 2 done: filter/state.py (NavState, error-state slices, inject), filter/params.py (FilterConfig YAML + FilterParams tensors from calibration), filter/learned.py (HeadOutputs.neutral), configs/filter/baseline.yaml.

## Summary

(written when Phase 3 completes or stops)
