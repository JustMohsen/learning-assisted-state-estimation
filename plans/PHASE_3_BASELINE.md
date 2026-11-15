# Phase 3 Plan: Analytical Baseline (Method A)

*Detailed plan for Phase 3 of [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md). **Status: IN PROGRESS** (the user asked in the morning of 15 November to finish Phase 3 and review it afterwards). Design choices remain proposals for review. Builds on the Phase 1 data foundation and the Phase 2 simulator and harness. The prototype from branch `phase3-prototype` has been restored onto `main`; its lessons are in section 7.*

## 1. Goal and exit criterion

A correct, consistent, fairly tuned **error-state Kalman filter (ESKF)** using IMU, FOG, wheels and consumer GPS, written once in PyTorch (float64, batch dimension), following [CONVENTIONS.md](../CONVENTIONS.md). Every later phase builds on it, and it is comparator A.

**Exit:**
- every validation check in section 6 passes;
- the filter is tuned on training days and scored on validation days with the Phase 2 harness;
- compute measurements are recorded.

## 2. Design

### 2.1 State (CONVENTIONS section 5)
- **Nominal state:** IMU position `p_w`, velocity `v_w`, attitude `q_wb`, accelerometer bias `b_acc`, gyro bias `b_gyr`, FOG bias `b_fog`.
- **Error state:** 16 coordinates `[dp, dv, dtheta, db_acc, db_gyr, db_fog]`; attitude error is right-multiplicative (body axes).

### 2.2 Propagation (at IMU events)
- **Interval:** `[t_k, t_{k+1}]` between consecutive valid IMU samples, using the average of corrected samples k and k+1 (midpoint), as in the validated strapdown integrator.
- **FOG fusion:** the FOG sensitive axis is body z (Phase 1: mount aligned).
  - The FOG increment over the interval is the difference between the latest FOG samples at or before `t_{k+1}` and `t_k`, sign-corrected (−1), minus `b_fog · dt`.
  - **This is causal, and its timing errors telescope:** the summed increments equal the total FOG angle change exactly.
  - The body-z rotation combines FOG and IMU gyro z with inverse-variance weights. With the configured noise this is about 100% FOG. If an interval has no FOG data, the IMU gyro z is used (fallback).
- **Mean:**
  - attitude `C ← C · Exp(φ)` with `φ = [ω̄_x dt, ω̄_y dt, Δθ_z]`;
  - acceleration `a = C_mid f̄ + g` (NED gravity, Phase 2 constant);
  - `p ← p + v dt + ½ a dt²`, `v ← v + a dt`.
- **Covariance:** first-order discrete error dynamics `F` and `P ← F P Fᵀ + Q`. Q comes from continuous noise densities: accelerometer and gyro white noise, FOG angle random walk, and bias random walks (configurable).
  - **Approximation (disclosed):** FOG angle noise is treated as white rate noise. Increment errors are actually anti-correlated and telescope.

### 2.3 Updates
- **Wheels (forward speed):**
  - measurement `z = (left + right) / 2`;
  - prediction `h = s_w · e_xᵀ (Cᵀ v − ω × l_imu)`, the forward speed at the axle centre with the IMU lever arm. Here `s_w` is the wheel speed scale, fixed at the Phase 1 training median of 1.054 (a reversible choice; see the questions for you).
  - scalar update with chi-square gating.
- **GPS (horizontal antenna position):**
  - measurement `z = gnss_ned[:2]`;
  - prediction `h = (p + C (l_gps − l_imu))[:2]`;
  - 2-dof chi-square gate. Held (repeated) fixes are skipped, since they carry no new information.
  - **Recovery rule:** after repeated rejections spanning over 30 s, inflate the position covariance so that genuine fixes can be re-acquired. This guards against permanent lock-out after divergence.
- **All updates:** analytical gain via linear solves, Joseph-form covariance, nominal injection, and reset of the attitude-error covariance.
- **Not in the baseline:** lateral no-slip constraint (scope stage 3), learned components (Phase 5), wheel-scale state (pending your decision).

### 2.4 Initialisation and output
- **Common initial-pose prior:** reference position and attitude at the first valid reference time (allowed by the scope).
- **Velocity:** from the wheel forward speed along body x (causal sensor data).
- **Biases:** zero, with configurable prior σ. No hidden truth is used.
- **Output:** a `NavEstimate` at each propagation step, transformed to the body reference point (axle centre), with a 5×5 covariance from the error-state Jacobians.

## 3. Code layout (`src/lase/filter/`)

| Module | Content |
|---|---|
| `so3.py` | Torch SO(3) and quaternion operations (Exp, Log, skew, quaternion products), batched, float64 |
| `state.py` | `NavState` (nominal), error-state index constants, initial covariance |
| `params.py` | `FilterConfig` (noise densities, priors, gates, calibration constants) |
| `propagation.py` | `propagate(state, P, interval, params, learned)` |
| `updates.py` | `update_wheel`, `update_gnss`, `UpdateDiagnostics`, Joseph form, injection and reset |
| `runner.py` | Event-stream loop: initialisation, propagation, updates, outputs as `NavEstimate`, diagnostics |
| `learned.py` | `HeadOutputs` with neutral defaults (CONVENTIONS 6.3), so the interface exists from the start |

## 4. Work items

1. Torch SO(3)/quaternion utilities + tests (round trips; against SciPy).
2. State, config and `HeadOutputs` (neutral); initial covariance.
3. Propagation (mean + covariance) + tests (matches the NumPy strapdown on clean simulated data; Jacobians vs autograd).
4. Wheel and GPS updates (Jacobians vs autograd; linear case vs batch weighted least squares).
5. Runner with initialisation and outputs; compute check (time per step).
6. Simulation validation: clean (tracks truth), nominal (bias convergence, NEES), slip, gps_degraded.
7. Real-data validation: oracle residuals (empirical R), NIS, comparison with GPS and odometry baselines, injected outages.
8. Tuning Q/R on training days; validation-day scores; report.

## 5. Compute

Measure the per-event cost in the Python loop. The baseline needs only forward runs: about 230k IMU steps per 85-minute session. If too slow, consider vectorising across sessions (batch) and `torch.compile`. Phase 4 training needs forward and backward passes and decides its own strategy from these numbers.

## 6. Validation (from DEVELOPMENT_PLAN section 7)

| Check | Independent reference |
|---|---|
| Jacobians F, H | `torch.autograd` Jacobians of the nonlinear models |
| Propagation mean | NumPy strapdown integrator on clean simulated data (agreement to numerical precision) |
| FOG handling | FOG-only heading vs reference heading over minutes (real data) |
| Propagation covariance | Monte Carlo over noisy simulated IMU data |
| Observation models | Oracle-state residuals on real data (reference state in, residual statistics out) |
| Update algebra | Linear-Gaussian case vs batch weighted least squares |
| Full filter, simulation | Bias convergence to injected values; NEES within chi-square bounds |
| Full filter, real data | NIS consistency; beats raw GPS; with GPS removed, within the range of NCLT odometry; outage error growth vs the Phase 2 reference points |

## 7. Lessons from the prototype (branch `phase3-prototype`, not merged)

The prototype implemented sections 2–3 and part of section 6. What it showed:

| Topic | Finding | Proposed plan change |
|---|---|---|
| Update timing | Applying wheel/GPS updates to the state of the previous IMU sample (up to ~20 ms stale) corrupted the gyro-z bias. The cause is the large, fast lever-arm term: the IMU sits 0.71 m above the axle on a Segway pitching ±20°. Result: **50° heading drift in 300 s on clean data**. | **Exact-time updates:** buffer each measurement until the next IMU sample, split the interval at measurement times, and interpolate IMU (and FOG) values. Costs ≤ 1 IMU interval of latency and uses no future data. |
| FOG increments | Re-evaluating the FOG angle at a boundary after a new FOG sample had arrived broke telescoping, and the error leaked into the FOG and gyro-z bias estimates | Reuse the boundary value, so increments sum exactly to the total FOG angle change |
| Transition matrix | First-order shortcuts (approximate right Jacobian; missing bias effect through the midpoint attitude) gave errors of ~5e-7 against autograd | Use the exact right Jacobian and include all bias terms; then F equals autograd to 1e-8 |
| Verification | The mean equalled the NumPy strapdown to 1e-9. Wheel/GPS H equalled autograd to 1e-10. Sequential updates equalled batch WLS to 1e-10. Monte Carlo covariance ratios were 0.88–1.15 (400 runs). | Keep these as the Phase 3 unit and validation tests |
| Simulation accuracy (17-min session) | Clean: 2 cm position, 0.1° heading. Matched-noise scenario: **NEES 2.4, 95% coverage 0.94, NIS wheel 1.00, NIS GPS 2.02**, i.e. consistent. Nominal (correlated GPS): 3.7 m RMSE but **NEES 133** (overconfident). | The filter is consistent when its noise model is right. Correlated GPS error needs an explicit decision: a GPS error state, R inflation or decimation, or leaving it to the learned multipliers. |
| Slip scenario | Without slip handling, heading RMSE rose to 3.8° and the gyro-z bias estimate was corrupted (7e-3 rad/s) | Slip robustness (gating, variance inflation, gyro-bias protection) belongs in the baseline design discussion |
| Compute | ~0.95 ms per IMU step including updates (B=1, CPU, Python loop): ~4 min per 85-minute session | A forward-only baseline is fine; Phase 4 training needs batching or compilation, so measure early |

### Decisions to make together

1. Wheel speed scale: fixed (1.054 training median), estimated as a slow state, or left to learning.
2. Correlated GPS error: add a 2-D Gauss–Markov GPS error state, or handle it in R (inflation or decimation).
3. Slip robustness in the baseline: chi-square gating only, or robust variance inflation and bias protection.
4. Exact-time update scheme (above): adopt?
5. Initialisation: reference pose + wheel velocity + zero biases, or stationary bias priors where available.
6. Tuning method on training days: grid/coordinate search on a few noise scalars, or learned constants (which overlaps with Phase 4's method B).
