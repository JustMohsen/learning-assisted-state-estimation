# Phase 4 pilot: learnings and where we stand

*15 November 2026. Companion to the pilot results ([README.md](README.md)) and the plan ([plans/PHASE_4_TRAINING.md](../../plans/PHASE_4_TRAINING.md)).*

## 1. Summary

- **Goal of the pilot:** a minimal working method B, trained and compared with baseline A (tag `baseline-a-pilot`) within 2–3 hours.
- **Outcome:** done in about 1 h 45 min from the start of coding. The training machinery works end to end and its gradients are verified.
- **Result:** B (learned GPS, wheel and lateral σ) is **worse than A** on the clean validation day: position median 5.38 vs 4.41 m, p95 10.6 vs 6.96 m, 120 s outage error 10.7 vs 6.1 m, position NEES 105 vs 45.
- **Main lesson:** training on short chunks warm-started from A rewards trusting the measurements more. The cost of that overconfidence shows only over minutes, so the objective cannot see it. Before B can beat a full-replay-tuned A, training has to see long horizons and select on full replays.

## 2. What was built

| Piece | Where | Status |
|---|---|---|
| Training-mode loop: `run_filter`'s event loop without NumPy on gradient tensors, a gradient switch, warm-start snapshots | `src/lase/training/loop.py` | Bitwise equal to `run_filter` with neutral outputs |
| Pilot method B: 3 σ as anchored deviations through the existing variance multipliers; Huber task loss | `src/lase/training/pilot.py` | Autograd = finite differences on real chunks |
| Pilot pipeline: cache A snapshots → train → evaluate on validation days → compare | `scripts/phase4_pilot.py` | Ran to completion |
| Unit tests: equivalence, warm-start continuation, gradients (with and without burn-in gradient) | `tests/unit/test_training_loop.py` | Pass (slow: ~85 s, open item O9) |
| `training` section (incl. `pilot`) | `configs/parameters.yaml` | In place |
| Autograd-safe lateral variance scaling (one line, identical numbers) | `src/lase/filter/updates.py` | Open item O2 (Phase 3 code) |

## 3. Learnings

### L1. The filter was already close to differentiable

Propagation and all updates were batched PyTorch functions, and the `HeadOutputs` hook (variance multipliers) already gave a way to inject learned values without touching the filter. Only two things broke the gradient graph:
- the runner's logging, which converts tensors to NumPy;
- one in-place operation in the lateral update (`R[:, 0, 0] = R[:, 0, 0] * mult`).

A copy of the event loop with those fixed reproduced `run_filter` exactly (difference 0.0 in state and covariance).

### L2. A no-gradient burn-in throws away most of the signal

The plan (D3) had a 20 s burn-in without gradient, then a 10 s loss window. On a real chunk, the GPS-σ gradient was:

| Gradient through | GPS σ | Wheel σ | Lateral σ |
|---|---|---|---|
| Loss window only | −0.009 | 0.054 | 0.156 |
| Whole 30 s chunk | 0.290 | 0.009 | 0.125 |

A noise level acts through the covariance, which builds up over the burn-in. A loss-window-only gradient sees the wrong (even sign-flipped) slope. The pilot therefore backpropagated through the whole chunk.

### L3. Measured cost

| Item | Measured |
|---|---|
| Filter forward | ~1.8 ms per IMU step (~50 Hz data), single thread |
| One 30 s chunk, gradient through the burn-in | ~20 s (vs ~3 s with a no-gradient burn-in) |
| One optimiser step (18 chunks, 9 processes) | ~80 s |
| Snapshot cache, 9 training days in parallel | ~22 min (with another job on the machine) |
| 40 training steps | ~55 min |
| Validation replay, 2 days × 2 modes in parallel | ~12 min |

Full-session or many-minute horizons will be considerably more expensive. The cloud cost estimate (D3) should be based on the long-horizon design, not on the pilot's 30 s chunks.

### L4. A short-horizon objective biases the noise toward overconfidence

B lowered all three σ (GPS 5.0 → 3.64 m, wheel 0.15 → 0.064 m/s, lateral 0.20 → 0.086 m/s). On a full validation session this made the filter overconfident:

| 2013-01-10, GPS mode | A | B | Ideal |
|---|---|---|---|
| Wheel NIS | 0.39 | 2.28 | 1 |
| GPS NIS | 3.37 | 7.49 | 2 |
| Position NEES | 46 | 131 | 3 |

Two mechanisms:
- **Horizon:** within 30 s, following the measurements closely pays off. Time-correlated wheel and GPS errors accepted as truth, a covariance that shrinks too far, and later fixes gated out cost accuracy only over minutes.
- **Warm start:** every chunk starts from A's larger covariance, so training never experiences the smaller covariance B's values settle to.

This extends D7: with time-correlated errors, even the measurement noise is effectively a slow parameter.

### L5. A's tuned values encode compensation for correlated errors

A's σ values were tuned on full replays and are deliberately inflated: GPS 5 m against 1.44 m white noise from the Phase 2 fit, and wheel NIS 0.39. That inflation compensates for errors the filter does not model as correlated. A learner that sees only short windows undoes it. **A full-replay-tuned A is a strong baseline; B has to be judged and selected on the same horizon.**

### L6. The anchor penalty was too weak as scaled

Dividing the penalty by the number of chunks (1,536) made it negligible: δ reached −0.85, beyond the prior width of 0.7, at almost no cost. The scaling (open item O7) needs a decision. It matters most when the data objective is biased, as here.

### L7. Selection on full replays is not optional

D6 planned early stopping on validation replays, and the pilot skipped it. With it, training would have stopped near δ = 0, i.e. B ≈ A. A full-replay check (training days for selection, validation days for confirmation) is the guard against a biased training objective.

### L8. The per-step training loss is not a progress signal

Each step draws 18 random chunks, so the loss jumped between 1.1 and 2.8 with no visible trend. Progress needs a **fixed set of monitoring chunks** and periodic full-replay metrics.

### L9. Process lessons

- Two Claude sessions share one git index: files staged in one were swept into the other's commit. Stage and commit in a single command.
- Parallel jobs from the two sessions competed for the 10 cores, roughly halving the speed of both.

## 4. Where we stand

### Work order (plan section 3)

| Step | Status |
|---|---|
| 1. Training-mode filter | **Partial:** separate copy of the loop, equivalent to `run_filter`; not merged (O3); not yet checked for each state layout |
| 2. Cached operation schedules | Not started (pilot used snapshot caches only) |
| 3. Batched masked runner | Not started (pilot: one process per session) |
| 4. Losses and gradient checks | **Partial:** task loss and finite-difference checks done; innovation NLL, formal gradcheck and loss scan not done |
| 5. Simulation recovery | Not started |
| 6. B stage 1 on real data | **Pilot only:** 3 of the D1 noise levels; result worse than A |
| 6a. Slow group (D7) | Not started |
| 7, 7a. Stage 2 constants, wheel-scale diagnostic | Not started |
| 8. Evaluate B vs A, report | Pilot comparison done |

### Open items

O1–O9 in plan section 7: gradient through the burn-in, the Phase 3 code change, merging the loop, the learned set, graph-node loss, no batched runner, anchor penalty scaling, pilot data and schedule, test runtime.

New from the result:

| # | Item |
|---|---|
| O10 | Horizon: carried-state chronological streams and/or longer loss windows |
| O11 | Full-replay selection and a fixed monitoring chunk set |
| O12 | Whether measurement noise should be grouped with the slow parameters (D7) while errors are modelled as white |

## 5. Recommended next steps (for discussion)

1. **Full-replay selection first** (cheap): evaluate a few δ values along the pilot's path on training-day replays. This confirms that the training objective and the full-session score disagree, and shows how far the optimum lies from δ = 0.
2. **Carried-state streams** (D3.4): each training day as consecutive chunks, so the covariance is B's own and errors accumulate. Then rerun the 3-σ pilot and see whether the bias toward small σ disappears.
3. **Simulation recovery** (D5 row 4): with white noise the method should recover the injected σ. If it does, the real-data bias comes from horizon and correlated errors, not the code.
4. **Longer loss windows** (1–5 min) with gradient checkpointing, costed before running.
5. **Small innovation-NLL weight** and a **stronger anchor**, as stabilisers. Treat them with care while A itself is overconfident (R8 pending).

## 6. Decisions needed

- Which of the next steps to take, and in what order.
- The anchor penalty scaling (O7).
- Whether to keep gradient through the burn-in as the default (O1).
- The Phase 3 one-line change in `updates.py` (O2): accept it into Phase 3, or route around it.
