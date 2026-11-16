# Phase 4 pilot: method B (3 learned noise levels) vs baseline A

*15 November 2026. Code: `scripts/phase4_pilot.py`, `src/lase/training/`. Baseline: tag `baseline-a-pilot` (f357fea). Plan: [plans/PHASE_4_TRAINING.md](../../plans/PHASE_4_TRAINING.md) section 7.*

## Setup

- **Learned:** GPS σ, wheel σ, lateral-constraint σ, as anchored deviations σ = anchor·exp(δ), through the filter's variance multipliers. B starts exactly at A.
- **Training:** 9 training days (1,536 chunks); each chunk warm-starts from A's cached state and covariance, then runs 20 s burn-in + 10 s loss window, with the gradient through the whole chunk. Huber task loss on position, heading and velocity against the dense reference. AdamW, 40 steps × 18 chunks, lr 0.05 (5 warm-up steps, cosine decay). About 55 min on 9 cores.
- **Evaluation:** validation days through `run_session` (same path as A), with a `Parameters` snapshot. Test days untouched.
- **Checks passed:** with neutral outputs the training loop is bitwise equal to `run_filter`; autograd matches finite differences (real chunks and unit tests).

## Learned values

| | A (anchor) | B | δ |
|---|---|---|---|
| GPS σ | 5.0 m | 3.64 m | −0.32 |
| Wheel σ | 0.15 m/s | 0.064 m/s | −0.85 |
| Lateral σ | 0.20 m/s | 0.086 m/s | −0.84 |

## Result: outages mode (`comparison_outages.csv`)

| Day | Method | Pos. median / p95 (m) | Heading RMSE | 60 s drift (nodes) | Outage 30 / 60 / 120 s (m) | Position NEES |
|---|---|---|---|---|---|---|
| 2013-01-10 (clean) | A | **4.41 / 6.96** | **1.29°** | **2.59 m** | **5.60 / 6.11 / 6.14** | **45** |
| | B pilot | 5.38 / 10.6 | 1.44° | 4.07 m | 6.58 / 7.34 / 10.7 | 105 |
| 2013-02-23 (median only) | A | 4.27 | | | | |
| | B pilot | 4.28 | | | | |

On 2013-02-23 B's p95 is lower (52.6 vs 67.8 m), but that day is dominated by the re-acquisition failure: a side effect, not a fix. GPS mode shows the same picture (2013-01-10: median 4.87 vs 3.98 m, p95 9.57 vs 6.32 m).

**B is worse than A on every metric on the clean day.**

## Why

B learned to trust every measurement more. On full sessions that makes the filter overconfident:

| 2013-01-10, GPS mode | A | B | Ideal |
|---|---|---|---|
| Wheel NIS (mean) | 0.39 | 2.28 | 1 |
| GPS NIS (mean) | 3.37 | 7.49 | 2 |
| Lateral NIS (mean) | 0.11 | 0.35 | 1 |
| Position NEES | 46 | 131 | 3 |

The training objective cannot see that cost:
1. **Horizon mismatch.** A 30 s chunk rewards tracking the measurements tightly. The cost of overconfidence (time-correlated wheel and GPS errors accepted as truth, a covariance that shrinks too far, later GPS fixes gated out) builds up over minutes. This is the same problem as the slow parameters (D7), now seen for the measurement noise.
2. **Warm start from A's covariance.** Every chunk starts with A's larger covariance, not the smaller one B's values would settle to, so 30 s chunks never show B's own steady state.
3. **Weak anchor.** The penalty divided by 1,536 chunks barely resists (δ ≈ −0.85, beyond the 0.7 prior width).
4. **No selection on full replays.** D6's early stopping on validation replays was skipped in the pilot. With it, training would have stopped near δ = 0 (B ≈ A).

## What the pilot established

- The machinery works end to end: differentiable loop, verified gradients, parallel training, evaluation on the same path as A.
- Short-chunk training of measurement noise from A's warm starts is biased toward overconfidence. Training must see longer horizons before B can beat a full-replay-tuned A.

## Candidate next steps (for discussion)

1. **Chronological streams with carried state** (D3.4): chunks continue each other, so the covariance is B's own and errors accumulate over minutes; or refresh the cache with B's current values every few steps.
2. **Longer loss windows** (1–5 min) with gradient checkpointing.
3. **Selection on full replays** (D6 early stopping) as a guard.
4. **Innovation NLL (α > 0)** to push towards consistency. Note A's position NEES is already ~45 (R8 pending), so treat this with care.
5. **A stronger anchor**: penalty not divided by the number of chunks, or a smaller σ_anchor.
