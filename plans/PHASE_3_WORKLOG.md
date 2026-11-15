# Phase 3 Worklog

Execution log for [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md). **Phase 4 must not be started before user review.**

## Decisions made

## Steps

- [ ] 1. Torch SO(3)/quaternion utilities + tests
- [ ] 2. State, FilterConfig, HeadOutputs (neutral), initial covariance
- [ ] 3. Propagation (mean + covariance) + tests (vs strapdown; Jacobians vs autograd)
- [ ] 4. Wheel and GPS updates + tests (Jacobians vs autograd; linear case vs WLS)
- [ ] 5. Runner (initialisation, event loop, NavEstimate output, diagnostics) + compute check
- [ ] 6. Simulation validation (clean, nominal, slip, gps_degraded): tracking, bias convergence, NEES
- [ ] 7. Real-data validation: oracle residuals, NIS, baselines comparison, injected outages
- [ ] 8. Tuning on training days; validation-day scores; Phase 3 report and summary

## Questions for the user

## Log

- 2026-11-15 02:05: Phase 3 started after Phase 2 completed: plan written (plans/PHASE_3_BASELINE.md).

## Summary

(written when Phase 3 completes or stops)
