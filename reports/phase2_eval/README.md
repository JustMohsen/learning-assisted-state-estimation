# Phase 2 Report: Simulator and Evaluation Harness

*15 November 2026. Plan: [PHASE_2_SIM_EVAL.md](../../plans/PHASE_2_SIM_EVAL.md). Execution log and decisions: [PHASE_2_WORKLOG.md](../../plans/PHASE_2_WORKLOG.md).*

Reproduce:

```bash
uv run pytest tests/validation                       # simulator S1-S3 (3 sessions), realism, harness E1-E2, Phase 1 checks
uv run python scripts/evaluate_baselines.py          # baseline metrics, injected and natural outage curves, figures
uv run python scripts/fit_sim_errors.py              # sim_fit.json: error models fitted on training days
uv run python scripts/validate_sim_realism.py realistic   # sim_realism_realistic.csv: real vs simulated statistics
```

## Simulator (`lase.sim`)

A real session's reference trajectory becomes a smooth spline truth: 100 Hz, 2 Hz zero-phase low-pass, quintic splines. Sensors are generated **at the session's real timestamps**, with known injected errors:
- IMU: lever arm, biases with random walk, noise, optional scale errors;
- FOG: released sign and wrapping, bias, noise;
- wheels: scale, track width, noise, slip episodes;
- GPS: correlated (Gauss–Markov) error, noise bursts, outliers, outages, held fixes, fix-mode switching.

The output types are the same as for real data (`SensorStreams`, `Reference`), plus a `SimTruth` with every hidden quantity. Scenarios are in `configs/sim/`: `clean`, `biased`, `nominal`, `slip` and `gps_degraded`. A full session simulates in about 0.2–1.3 s.

| Check | Result (2012-01-22, nominal) | Status |
|---|---|---|
| S1 independent strapdown on clean data | 8.4 mm error per 10 s at 47 Hz; 0.53 mm at 4× rate. Convergence ratio 16.1, matching second order. | Pass |
| S2 Phase 1 checks recover injected values | GPS median 3.41 m (injected 3.41); wheel scale 1.0546 (1.054); raw track 0.5905 m (0.590); FOG/IMU scale −0.996/+0.996 with sub-ms lag | Pass |
| S3 noise statistics | Accelerometer 0.0201 (config 0.02); gyro 0.00299 (0.003) | Pass |

The heading-increment scale is 0.996 even with exact sensors, because Euler yaw rate differs from body z rate when the Segway pitches (up to about 20°). The real-data value of 0.998 is therefore consistent with a perfect scale.

## Evaluation harness (`lase.evaluation`)

| Component | Behaviour |
|---|---|
| `NavEstimate` | N/E position, N/E velocity, heading and optional 5×5 covariance per output time |
| Association | Causal: latest estimate at or before each reference time (staleness ≤ 200 ms). Uncovered times lower `coverage`. |
| Accuracy | Position RMSE, mean, median, p95, max; heading RMSE (wrapped); velocity RMSE. At graph nodes (primary) and on a 10 Hz dense grid; per session and pooled. |
| Drift | Planar relative pose error over 10 s and 60 s |
| GPS outages | Seeded 30/60/120 s windows where GPS was healthy, withheld from the inputs; error at fixed offsets after the outage starts |
| Consistency | NEES and NIS with chi-square coverage |
| Cross-checks | E1: position RMSE identical to `evo` APE. E2: the reference scored against itself gives 6 mm (nodes; causal association) and 0 (dense). |

## Baseline reference points (all 27 sessions)

Pooled at graph nodes. Full tables: `baselines_pooled.csv`, `baselines_by_session.csv`.

| Method | Split | Position median | Position p95 | Position RMSE | Heading RMSE |
|---|---|---|---|---|---|
| Raw GPS | train / val / test | 3.7 / 4.5 / 4.0 m | 15 / 19 / 43 m | 143 / 378 / 336 m | 34 / 38 / 42° (course) |
| NCLT odometry (no GPS, aligned at start) | train / val / test | 28 / 74 / 57 m | 114 / 221 / 148 m | 54 / 111 / 75 m | 12 / 25 / 15° |

GPS outages (232 injected windows; `outage_growth.png`, `outage_curves.json`):

| Time since outage start | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| GPS hold (no dead reckoning), median | 15 m | 38 m | 62 m | 109 m |
| Oracle-anchored NCLT odometry (diagnostic), median | 0.22 m | 0.55 m | 1.0 m | 2.0 m |
| Oracle-anchored NCLT odometry, p90 | 0.49 m | 1.3 m | 2.6 m | 4.5 m |

## Findings for Phase 3

- **Extreme GPS outliers flagged as valid:** 1.4% of samples are over 100 m off and 0.4% over 1 km (up to 5.9 km), in 15 of 27 sessions. This explains the raw-GPS RMSE. **Robust gating is mandatory.** Compare medians and p95 as well as RMSE.
- **Dead-reckoning target:** wheels plus FOG from a perfect start drift about 1 m per minute (median). A well-tuned baseline should come close to this during outages, and the learned components aim to improve on it.
- **Without GPS, global error accumulates:** NCLT's own odometry reaches tens of metres over a session. The value of GPS is bounding that drift; the value of the filter is bridging gaps and rejecting outliers.

## Hardening (after the initial Phase 2 completion)

### Error models fitted to real data (`sim_fit.json`, 22 training days)

| Quantity | Fitted value | Note |
|---|---|---|
| GPS inlier error | 1.44 m "white" + 2.53 m Gauss–Markov, τ = 55 s | Close to the earlier defaults (1.5 / 2.5 / 60 s) |
| GPS errors > 10 m | Episodic: ~22–27 runs/h, median 2.2 s, p90 47 s; only 32% isolated fixes | Modelled as multipath episodes |
| Gross GPS errors > 50 m | ~3.4 episodes/h, median 14 s, offset median 81 m, p90 1 km, max 5 km | Modelled as gross episodes |
| GPS fix mode | Mode 2 for 44% of fixes, alternating almost every fix; accuracy 2.94 vs 2.78 m | No separate mode-2 noise needed |
| Held GPS fixes | 0.7–1% | |
| Wheel mean-speed residual | Robust std 0.035 m/s (includes reference error) | Upper bound on noise |
| Slip-like wheel excursions | Strongly session-dependent: median 0.019 per moving minute, pooled 0.09 | |
| IMU at rest | Dominated by self-balancing motion; quietest axes bound noise at 0.002 m/s² / 0.0025 rad/s | Not identifiable; nominal values kept |

The new `realistic` scenario uses these values. Two new simulator mechanisms support it: episodic GPS errors and a separate yaw low-pass cutoff.

### Realism check (`sim_realism_realistic.csv`; six training sessions, simulated twins vs real, ratio of session medians)

| Matches well (0.9–1.1×) | Mismatch | Cause |
|---|---|---|
| GPS median, inlier σ, time > 10 m, gross-episode rate, held fixes, fix-mode fraction, wheel residual and excursions, yaw-rate content, forward acceleration (0.87×) | > 10 m runs 1.7×; lateral/vertical acceleration 1.8×/2.4×; roll rate 0.18× | Smoothing position and attitude separately breaks the physical coupling: the Segway rolls about its wheel contact points, and the 0.71 m IMU lever arm amplifies spurious roll content. Position/roll/pitch at 1 Hz with yaw at 5 Hz is the best compromise found. |

### Natural GPS outages (`natural_outage_curves.json`, `natural_outage_growth.png`)

84 natural outages over 5 s inside the reference span: median 122 s, p90 218 s, max 837 s. These are **longer than the injected windows**, and GPS is already degraded when they start. Holding the last fix is 46 m off at outage start.

| Time since natural outage start | 10 s | 60 s | 120 s | 300 s |
|---|---|---|---|---|
| Oracle-anchored NCLT odometry, median | 0.26 m | 1.4 m | 2.6 m | 4.6 m |

### Standard entry point

`lase.evaluation.report.evaluate_estimate` produces node/dense metrics, RPE, injected- and natural-outage curves, and NEES for any `NavEstimate`. Edge cases (empty estimates, NaN fields, empty windows) are tested and raise no warnings.
