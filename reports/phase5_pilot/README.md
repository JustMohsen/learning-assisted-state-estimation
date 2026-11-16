# Phase 5 pilot: method C (GRU noise multipliers) vs A and B

*15 November 2026. Code: `scripts/phase5_pilot.py`, `src/lase/training/pilot_c.py`. Baseline A: tag `baseline-a-pilot`. B: [Phase 4 pilot](../phase4_pilot/README.md) ([learnings](../phase4_pilot/LEARNINGS.md)).*

## Setup

- **Model:** a GRU (hidden 32, one layer) over 16 causal, sensor-only features on a 10 Hz grid: IMU means and vibration, wheel speeds, wheel and FOG turn rates, GPS recency, fix mode, and GPS-vs-wheel speed consistency. A linear head gives per-step variance multipliers for GPS, wheel and lateral constraint, bounded to [¼, 4]. It starts at zero, so the untrained network is exactly A.
- **Training:** identical to the B pilot (same 9 training days, A warm-start snapshots, 20 s burn-in + 10 s loss window, gradient through the whole chunk, same Huber loss), so C vs B isolates the effect of context. Adam lr 3e-3. **Stopped at step 23 of 120** (24 steps × 9 chunks) to keep to about an hour; one step stalled for 25 min.
- **Evaluation:** validation days through `run_session` (the same path as A and B), with C's outputs passed to `run_filter`.
- **Checks:** untrained C equals A (chunk loss and `run_session` segment); head gradient = finite difference.

## Result

Outages mode (`comparison.csv`):

| Day | Method | Pos. median / p95 (m) | Heading RMSE | 60 s drift | Outage 30 / 60 / 120 s (m) | NEES |
|---|---|---|---|---|---|---|
| 2013-01-10 (clean) | A | **4.41 / 6.96** | 1.29° | **2.59** | **5.60 / 6.11 / 6.14** | **45** |
| | B pilot | 5.38 / 10.6 | 1.44° | 4.07 | 6.58 / 7.34 / 10.7 | 105 |
| | C pilot | 4.84 / 7.52 | **1.28°** | 3.16 | 5.96 / 6.48 / 7.55 | 68 |
| 2013-02-23 (median; GPS failure day) | A | 4.27 | 2.77° | | | |
| | B pilot | 4.28 | 2.83° | | | |
| | C pilot | **4.22** | **2.13°** | | | |

GPS mode, 2013-01-10: median 3.98 (A) / 4.87 (B) / 4.29 (C) m; heading 1.37° / 1.76° / 1.61°. 2013-02-23 heading: 2.24° / 2.99° / **1.84°**.

Consistency (GPS mode, 2013-01-10): wheel NIS 0.39 / 2.28 / 1.33; GPS NIS 3.4 / 7.5 / 4.5; position NEES 46 / 131 / 76.

## Reading

- **C beats B on every clean-day metric but does not beat A.** C moved in the same direction as B (median variance multipliers GPS 0.78, wheel 0.44, lateral 0.44, i.e. σ × 0.88 / 0.66 / 0.66), but less far. Context variation was small (about ±5–10% across chunks). Most of the C-vs-B gap is therefore probably *less training*, not context.
- **Possible context effect on 2013-02-23:** heading 2.13° vs 2.77° (A) in outages mode and 1.84° vs 2.24° in GPS mode, the day with the GPS failure. A constant multiplier made B worse there (2.83°), so the difference is not explained by trusting the measurements more. This is one day and a partially trained network: a lead to check, not a result.
- The training objective's bias (B pilot, L4) still applies: C cannot beat A until training sees long horizons.

## Learnings

1. **The Phase 4 machinery carried over directly.** The training loop, A snapshots, loss and evaluation path were reused unchanged; C took about 40 min to build and verify. The `learned_fn` / `HeadOutputs` interface is the right seam between filter and network.
2. **Under the same objective, C drifts the same way as B.** The network first learned a near-uniform shift towards trusting the measurements (the short-horizon bias, Phase 4 learnings L4); context-dependence came second and stayed small. Fixing the objective (long horizons, full-replay selection) comes before judging context.
3. **Partial training confounds C vs B.** C moved less far than B, so it is closer to A for that reason alone. A fair comparison needs a matched-constant control or equal training progress.
4. **A lead worth following:** better heading on the GPS-failure day (2013-02-23), where a constant multiplier (B) made heading worse.
5. **Robustness:** one optimiser step stalled for 25 min. Worker futures need timeouts (skip a slow chunk rather than wait).
6. **Evaluation hook:** `run_session` should take a `learned_fn` (or a network) instead of being patched (C3).

## To disentangle (next)

1. **Matched-constant control:** evaluate B with C's median multipliers as constants (δ = ½ ln m: GPS −0.12, wheel −0.41, lateral −0.41). If C ≈ that control on 2013-01-10 but better on 2013-02-23, context is doing something. Cost: one validation run (~12 min).
2. Inspect C's multipliers over time on 2013-02-23 around the GPS failure.

## Open items

| # | Item |
|---|---|
| C1 | Training incomplete (24 of 120 steps); a step stalled for 25 min |
| C2 | Same short-horizon objective as B (see Phase 4 learnings L4, O10–O11) |
| C3 | Evaluation patches `experiment.run_filter` to pass `learned_fn`; `run_session` should accept it directly (Phase 3 code) |
| C4 | Reconcile `pilot_c.PilotNet` with the `lase.learning.context_net` design skeleton |
| C5 | Feature set: the satellite count is always 0 in this data (inert feature); no filter-state inputs yet |
