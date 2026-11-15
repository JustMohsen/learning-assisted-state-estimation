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

## Steps

- [x] 1. Truth trajectory: resample, zero-phase low-pass, splines; exact derivatives and body rates
- [x] 2. Sensor models: IMU (lever arm), FOG (released convention), wheels (scale, track, slip), GPS (Gauss–Markov, bursts, outliers, outages, held)
- [x] 3. Scenarios (`configs/sim/`), `SimTruth`, outputs as `SensorStreams` + `Reference`; `scripts/simulate.py`
- [ ] 4. Simulator validation S1–S3 (+ tests)
- [ ] 5. Evaluation core: `NavEstimate`, causal association, accuracy metrics (nodes + dense)
- [ ] 6. Outage injection, outage error-growth curves, relative pose error
- [ ] 7. Consistency statistics (NEES/NIS, coverage), plots
- [ ] 8. Baselines (raw GPS, NCLT odometry) + `evo` cross-check (E1–E2)
- [ ] 9. Phase 2 report + summary; update README status

## Questions for the user

## Log

- 2026-11-15 01:50: Phase 2 started: plan written (plans/PHASE_2_SIM_EVAL.md); run control in plans/AUTONOMOUS_RUN.md; wake-ups every 20 min + one-shot 04:30.
- 2026-11-15 01:55: Step 1 done: sim/trajectory.py (truth splines, exact derivatives, body rates, FOG body-z angle antiderivative).
- 2026-11-15 01:55: Step 2 done: sim/sensors.py (IMU with lever-arm acceleration, FOG released convention, wheels with scale/track/slip, GPS with Gauss-Markov, bursts, outliers, outages, held, mode Markov chain).
- 2026-11-15 01:55: Step 3 done: sim/config.py + 5 scenarios, sim/simulate.py (SimRun = SensorStreams + Reference from truth + SimTruth); 2013-01-10 simulates in ~0.2 s per scenario; 8 synthetic-circle unit tests (exact lever-arm specific force, wheel kinematics, FOG convention, outages, random-walk statistics).

## Summary

(written when Phase 2 completes)
