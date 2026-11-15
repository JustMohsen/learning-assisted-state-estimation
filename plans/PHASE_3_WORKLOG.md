# Phase 3 Worklog

Execution log for [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md). **Phase 4 must not be started before user review.**

## Decisions made

- **Filter numerics:** PyTorch float64 on CPU, batch-first; SO(3)/quaternion operations with Taylor branches at small angles so values and gradients stay finite (needed for Phase 4 training).
- **Default noise densities (to be tuned):** accelerometer 0.02 m/s^2/sqrt(Hz) (includes Segway vibration), gyro 0.002 rad/s/sqrt(Hz), FOG 2e-4 rad/sqrt(s), bias random walks 1e-4 / 1e-5 / 1e-7. Wheel sigma 0.1 m/s; GPS sigma 3 m; gates at 99.9%. Initial prior: position 0.5 m, velocity 0.3 m/s, attitude 0.02 rad, biases 0.1 m/s^2 / 0.01 rad/s / 1e-4 rad/s.
- **HeadOutputs interface implemented now with neutral defaults** (zero corrections, unit multipliers), so the baseline and learned methods share one code path from the start.
- **IMU/FOG fusion:** body-z rotation = inverse-variance combination of the FOG increment (sign-corrected, minus b_fog dt) and IMU gyro z; with default noise the FOG weight is ~0.99. If no FOG data covers the interval, the weight is 0 (IMU fallback). The FOG increment uses the latest FOG samples at or before each interval boundary: causal, and timing errors telescope.
- **Exact first-order transition:** F includes the gyro/FOG-bias effect through the midpoint attitude on velocity and position and uses the exact SO(3) right Jacobian; verified equal to autograd of the nonlinear propagation to 1e-8.
- **Disclosed approximation:** FOG angle noise is modelled as white rate noise (angle random walk) in Q; the true increment errors are anti-correlated and telescope. To revisit with the colored-noise derivation listed in the scope.
- **Wheel update:** z = (left + right)/2; h = s_w * forward speed of the axle centre (C^T v - omega x l_imu) + learned residual; the latest corrected body rate is an exogenous input (gyro-noise reuse is a disclosed approximation, as in the scope). s_w fixed at the calibration estimate 1.054 pending the user's decision.
- **GPS update:** horizontal antenna position h = (p + C (l_gps - l_imu))[:2], R = sigma^2 * learned multipliers (diagonal N/E). Rejected updates leave state and covariance untouched.
- **Covariance handling:** Joseph form plus first-order attitude reset Jacobian G = I - 0.5[dtheta]x after injection; symmetrisation for round-off only.

## Steps

- [x] 1. Torch SO(3)/quaternion utilities + tests
- [x] 2. State, FilterConfig, HeadOutputs (neutral), initial covariance
- [x] 3. Propagation (mean + covariance) + tests (vs strapdown; Jacobians vs autograd)
- [x] 4. Wheel and GPS updates + tests (Jacobians vs autograd; linear case vs WLS)
- [ ] 5. Runner (initialisation, event loop, NavEstimate output, diagnostics) + compute check
- [ ] 6. Simulation validation (clean, nominal, slip, gps_degraded): tracking, bias convergence, NEES
- [ ] 7. Real-data validation: oracle residuals, NIS, baselines comparison, injected outages
- [ ] 8. Tuning on training days; validation-day scores; Phase 3 report and summary

## Questions for the user

## Log

- 2026-11-15 02:05: Phase 3 started after Phase 2 completed: plan written (plans/PHASE_3_BASELINE.md).
- 2026-11-15 02:07: Step 1 done: filter/so3.py (skew, Exp/Log, quaternion product/exp/normalise, matrix<->quaternion, yaw) with tests against SciPy and gradient finiteness at zero.
- 2026-11-15 02:07: Step 2 done: filter/state.py (NavState, error-state slices, inject), filter/params.py (FilterConfig YAML + FilterParams tensors from calibration), filter/learned.py (HeadOutputs.neutral), configs/filter/baseline.yaml.
- 2026-11-15 02:09: Step 3 done: filter/propagation.py (midpoint mean, FOG fusion, F, Q, boxminus) and exact SO(3) right Jacobian. Tests: mean equals the NumPy strapdown to 1e-9 over 300 irregular steps (no FOG); F equals autograd Jacobian to 1e-8 with and without FOG.
- 2026-11-15 02:09: Note: ruff --unsafe-fixes rewrote 'array + [list]' into list unpacking in a test (RUF005) and broke it; unsafe fixes are no longer applied wholesale.
- 2026-11-15 02:11: Step 4 done: filter/updates.py (gated kalman_update, update_wheel, update_gnss). Tests: wheel/GPS H equal autograd to 1e-10; sequential linear updates equal batch WLS to 1e-10; 1 km GPS outlier rejected with unchanged state; wheel update reduces speed error.

## Summary

(written when Phase 3 completes or stops)
