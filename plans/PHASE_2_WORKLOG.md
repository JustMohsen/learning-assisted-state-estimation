# Phase 2 Worklog

Execution log for [PHASE_2_SIM_EVAL.md](./PHASE_2_SIM_EVAL.md). Run rules: [AUTONOMOUS_RUN.md](./AUTONOMOUS_RUN.md).

## Decisions made

## Steps

- [ ] 1. Truth trajectory: resample, zero-phase low-pass, splines; exact derivatives and body rates
- [ ] 2. Sensor models: IMU (lever arm), FOG (released convention), wheels (scale, track, slip), GPS (Gauss–Markov, bursts, outliers, outages, held)
- [ ] 3. Scenarios (`configs/sim/`), `SimTruth`, outputs as `SensorStreams` + `Reference`; `scripts/simulate.py`
- [ ] 4. Simulator validation S1–S3 (+ tests)
- [ ] 5. Evaluation core: `NavEstimate`, causal association, accuracy metrics (nodes + dense)
- [ ] 6. Outage injection, outage error-growth curves, relative pose error
- [ ] 7. Consistency statistics (NEES/NIS, coverage), plots
- [ ] 8. Baselines (raw GPS, NCLT odometry) + `evo` cross-check (E1–E2)
- [ ] 9. Phase 2 report + summary; update README status

## Questions for the user

## Log

- 2026-11-15 01:50: Phase 2 started: plan written (plans/PHASE_2_SIM_EVAL.md); run control in plans/AUTONOMOUS_RUN.md; wake-ups every 20 min + one-shot 04:30.

## Summary

(written when Phase 2 completes)
