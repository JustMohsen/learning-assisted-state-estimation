# Phase 3 Report: Analytical Baseline (Method A)

*15 November 2026. Plan: [PHASE_3_BASELINE.md](../../plans/PHASE_3_BASELINE.md). Execution log and decisions: [PHASE_3_WORKLOG.md](../../plans/PHASE_3_WORKLOG.md). Status: complete, **pending review with the user** (several design choices are proposals; see "Open questions").*

Reproduce:

```bash
uv run pytest tests/unit tests/validation              # 83 unit + 31 validation tests
uv run python scripts/sim_baseline.py baseline        # sim_scenarios_baseline.json
uv run python scripts/tune_baseline.py [1|2|3 ...]    # tuning_*.csv (training days only)
uv run python scripts/evaluate_baseline.py baseline   # validation_* (validation days only)
```

## 1. What was built

An error-state Kalman filter in PyTorch (float64, batch-first; `src/lase/filter/`), configured by [configs/parameters.yaml](../../configs/parameters.yaml) (all tunable values, commented).

| Part | Design |
|---|---|
| State | IMU position, velocity, attitude (quaternion), accelerometer/gyro/FOG biases, wheel speed scale. 17 error coordinates; the wheel scale is fixed by default. |
| Propagation | Midpoint IMU integration. Body-z rotation is the inverse-variance fusion of FOG and gyro z (FOG ≈ 100%). Exact first-order transition (F = autograd to 1e-8). Optional Earth-rate compensation (off; see 4.4). |
| Updates (gated, Joseph form) | Wheel forward speed (lever arm, scale); GPS horizontal antenna position at 1 Hz; **optional soft non-holonomic constraint** (axle lateral/vertical velocity ≈ 0; on in the tuned config). All applied at their exact acquisition times. |
| Robustness | 99.9% chi-square gates; GPS recovery (covariance inflation) after 300 s of rejections |
| Learned interface | `HeadOutputs` (bias drift, wheel residual, wheel/GPS variance multipliers), present with neutral values, so methods B/C reuse this code |
| Output | `NavEstimate` at the axle centre with a 5×5 covariance, scored only through `evaluate_estimate` |

## 2. Verification (independent references)

| Check | Result |
|---|---|
| Propagation mean vs NumPy strapdown | Equal to 1e-9 over 300 irregular steps |
| Transition F vs autograd | Equal to 1e-8 (with/without FOG and Earth rate) |
| Wheel / GPS / non-holonomic H vs autograd | Equal to 1e-10 |
| Sequential updates vs batch weighted least squares | Equal to 1e-10 |
| Propagated covariance vs Monte Carlo (400 runs) | Variance ratios 0.88–1.15 |
| Clean simulation (default config) | 2 cm position, < 0.2° heading |
| Matched-noise simulation (default config) | NEES 2.4 (ideal 2), NIS wheel 1.00, GPS 2.02 |
| Wheel-scale state on a synthetic circle | Converges from 1.054 to the true 1.08 (±0.003) |

## 3. Results

*Sections 3.1 and 3.3 show the **frozen pilot baseline** (tag `baseline-a-pilot`, commit f357fea): wheel scale from the date trend as a consider state, FOG bias prior 1e-4 rad/s and random walk 1e-6, white GPS (5 m, 1 Hz), grade vertical mode, lateral σ 0.2 m/s. This is the baseline the Phase 4 pilot compares against. Numbers for the baseline before the review revisions (e.g. 4.7–5.3 m validation median) are in the git history. Section 3.2 records the original tuning.*

### 3.1 Simulation, tuned config (`sim_scenarios_baseline.json`, 2013-01-10 trajectory)

| Scenario | Position median / p95 | Heading RMSE | NEES (ideal 2) |
|---|---|---|---|
| clean | 0.33 / 0.99 m | 0.21° | 1.4 |
| nominal (correlated GPS) | 1.15 / 3.15 m | 0.62° | 16 |
| slip | 1.15 / 3.72 m | 0.42° | 12.5 |
| gps_degraded | 1.98 / 5.38 m | 1.63° | 25.5 |
| realistic (fitted) | 1.30 / 2.81 m | 0.38° | 11.2 |

The tuned GPS settings (5 m, 1 Hz) trade a little clean-data accuracy for robustness to correlated and episodic GPS errors. With realistic errors the filter is still overconfident (NEES 11–25), because the time-correlated GPS error is not modelled.

### 3.2 Tuning on training days (`tuning_*.csv`; original tuning, before the review revisions)

There were 3 stages and 25 configurations: 156 runs of 30-minute segments on 6 training sessions, with injected outages. The optimum is flat. Chosen:
- **GPS sigma 5 m at 1 Hz**;
- **wheel sigma 0.15 m/s**;
- **accelerometer noise 0.05**;
- **non-holonomic constraint on**;
- **GPS recovery after 300 s**: 90 s accepted gross GPS episodes, causing a 112 m jump on 2012-02-04.

Training medians: position 3.1 m (median) / 5.7 m (p95), heading 1.36°, velocity 0.18 m/s, error at outage end 3.2 m.

### 3.3 Validation days (`validation_runs.csv`, figures `validation_*.png`)

Graph-node scoring; drift = 60 s relative pose error on graph nodes.

| Method (2013-01-10 / 2013-02-23) | Position median | Position p95 | Heading RMSE | Drift over 60 s | Position NEES (ideal 3) |
|---|---|---|---|---|---|
| **Baseline A, all sensors** | 4.0 / 4.3 m | 6.3 / **65.9 m** | 1.37 / 2.24° | 2.8 / 11.6 m | 46 / 263 |
| Baseline A, injected outages | 4.4 / 4.3 m | 7.0 / 67.8 m | 1.29 / 2.77° | 2.6 / 12.1 m | 45 / 264 |
| Baseline A, no GPS | 12.6 / 136 m | 29.3 / 366 m | 2.20 / **39.0°** | 3.7 / 5.1 m | 56 / 1834 |
| Baseline without non-holonomic constraint, all sensors | 5.6 / 44.7 m | 23.7 / 362 m | 7.5 / 30.0° | 11.9 / 25.0 m | 47 / 84 |
| Raw GPS (reference point) | 4.7 / 4.4 m | 11.3 / 21.4 m | 31 / 39° (course) | 37 / 385 m | |
| NCLT odometry, no GPS (reference point) | 11.9 / 90 m | 34.7 / 255 m | 3.5 / 27.2° | 2.3 / 2.7 m | |

**Injected GPS outages** (11 windows on the validation days; `validation_outage_growth.png`):

| Time since outage start | 0 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| Baseline A, median | 4.0 m | 4.5 m | 3.7 m | 6.1 m |
| Without non-holonomic constraint, median | 19.6 m | 34.7 m | 57.2 m | 63.9 m |
| Oracle-anchored NCLT odometry (diagnostic, perfect start), median | 0 m | 1.3 m | 2.5 m | 4.4 m |

**Reading:**
- With all sensors, the baseline beats raw GPS in median on both days and in p95 on 2013-01-10 (6.3 vs 11.3 m), and it provides heading and velocity. On 2013-02-23 the GPS re-acquisition failure (open problem 1) gives a p95 of 66 m.
- During outages, its error **grows** by about 2 m over 120 s (median 4.0 → 6.1 m), less than NCLT's own odometry from a perfect start (4.4 m). Its absolute error is higher because it starts from a GPS-level error of about 4 m.
- Without GPS, 2013-02-23 drifts 39° in heading (FOG drift, open problem 2).
- Position NEES of 46 and 263 (ideal 3): the filter is overconfident (consistency work R8 pending).

## 4. Findings and problems found (and fixed) on the way

1. **Updates must be exactly time-aligned.** Stale-state wheel updates plus the 0.71 m lever arm caused 50° heading drift on clean data.
2. **FOG increments must telescope,** and **FOG extrapolation needs a ≥ 20 ms baseline.** Real FOG samples sometimes arrive 0.1 ms apart, which produced one-sample heading spikes of up to 180° and corrupted the biases.
3. **FOG noise from the datasheet:** looser values let GPS errors drag the heading.
4. **Earth rotation: implemented but off.** Compensating made heading worse on training and validation days. The residual FOG drift has the opposite sign to an uncompensated Earth rate, so the released data appears already compensated.
5. **Lateral velocity is unobservable without the non-holonomic constraint.** That makes the scope-literal baseline drift by hundreds of metres.
6. **The wheel-scale state is not cleanly identifiable here.** Enabled, it drifted to 1.08–1.14 and doubled errors.

## 5. Open problems

1. **GPS re-acquisition after outages (2013-02-23: p95 66 m).** After a natural outage the grown uncertainty lets the first, grossly wrong fixes (about 67 m) pass the gate; the filter jumps, then rejects good GPS for about 6 minutes until the 300 s recovery. Candidates: require several mutually consistent fixes after an outage; robust (heavy-tailed) GPS updates; check GPS displacement against wheel odometry.
2. **Unexplained FOG drift:** about +2 to +4° per 10 min relative to the reference in every session, and 27° over 80 min on 2013-02-23 (where NCLT's own odometry shows the same). A wider FOG-bias prior made results worse.
3. **Overconfidence:** NEES ≫ 2 on real data and realistic simulation, from time-correlated GPS error. Options: a GPS error state, or leave it to the learned GPS multipliers (Phase 5).
4. **Compute:** about 1.9–2.7 ms per IMU step per process (Python loop, B = 1), about 3–4 minutes per session. Phase 4 training will need batching or compilation.

## 6. Open questions for the user

See [REVIEW_PHASES.md](../../plans/REVIEW_PHASES.md), section 5.
