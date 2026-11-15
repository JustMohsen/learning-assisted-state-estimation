# Training protocol — accepted starting point

14 November 2026. **Recommendations accepted as an adjustable starting point.** Sequence navigation supervision through the analytical ESKF/MSCKF, event-aware regularization, uncertainty-calibration assessment and fixed-weight inference are agreed. Split dates, stateful chunking, initial optimizer/loss settings, staged learning and validation/stopping recommendations below are accepted; details can be refined during implementation. No training throughput, memory use or results have been measured. Evaluation/comparisons and implementation/compute are accepted; rollout order is accepted; current discussion: [model learning and Gemma 4](./model-learning-and-gemma4.md).

## Dataset split and information contract

Recommend the **22 training / 2 validation / 3 test recording-day split** in Song et al.'s official NCLT implementation. It is a published implementation's split, not a universal official NCLT benchmark. Use its dates while retaining our own native asynchronous sensor processing; do not silently inherit its 1-Hz resampling, vendor-heading inputs or reference-derived coordinate offsets. [Official split source](https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/data/NCLT/preprocess.py).

```text
Training (22):
2012-01-08, 2012-01-15, 2012-01-22,
2012-02-02, 2012-02-04, 2012-02-05, 2012-02-12, 2012-02-18, 2012-02-19,
2012-03-17, 2012-03-25, 2012-03-31,
2012-04-29, 2012-05-11, 2012-05-26, 2012-06-15,
2012-08-04, 2012-08-20, 2012-09-28, 2012-10-28, 2012-11-17, 2012-12-01

Validation (2): 2013-01-10, 2013-02-23
Test (3):       2012-11-04, 2012-11-16, 2013-04-05
```

Freeze this manifest before fitting normalization, baseline noise or learned parameters. A new/unlisted recording must not silently join a split. Tune settings/checkpoints on validation only. Reserve the test recordings for final frozen-method comparisons, not model selection. The task remains cross-day transfer on the same campus, not demonstrated unseen-location generalization.

For bring-up, use short clips from one training day and a separate validation day, then expand to the full split as data/cache/runtime checks pass. All training data stays within the listed training days. This staging does not change the final held-out protocol or require downloading every image archive immediately.

Use the accepted controlled startup prior consistently with baselines. No reference-state replacement at each step/chunk, hidden true-bias initialization or reference-based recovery. Sensor-only startup remains a separately labeled mode. Cached features/masks/matches are causal frontend outputs; no test reference is an online quality feature. Fixed published camera calibration is allowed across days; learned normalization/error/noise fits use training data only.

## Stateful sequences and initial numerical settings

| Choice | Recommendation |
|---|---|
| Sampling | Two independent chronological recording streams per mini-batch initially; each owns its filter state/covariance and GRU memory. Shuffle recording selection/order, not time within a stream |
| Gradient history | Start with **10-second truncated-backpropagation chunks**, using actual sensor events. Carry numerical state/covariance/memory across contiguous chunks while detaching gradients |
| Memory tuning | If memory is excessive, reduce to one stream first; adjust chunk length after profiling. Investigate longer gradient histories if they improve bias/drift learning on validation |
| Chunk initialization | Begin from a valid recording startup or a causally generated roll-in/checkpoint. A random clip cannot silently receive the true full state or zeroed recurrent memory from an established run |
| Startup warm-up | Preserve the accepted causal baseline warm-up/ramp; choose its duration during implementation. Do not repeat it at every truncation boundary or silently omit startup errors from evaluation |
| Optimizer | **AdamW, learning rate 1e-4, weight decay 1e-5**, gradient-norm clipping at **1.0**, as tunable initial defaults |
| Precision | Accepted Float64 filter/covariance and Float32 learned module, with differentiable casts. Frozen frontend computations are cached rather than backpropagated |
| Reproducibility | One fixed seed for bring-up; target three seeds for the final pilot comparisons, with split, versions, weights, cache provenance and configuration recorded |

A 10-second training chunk is neither a 10-second inference delay nor a GRU-memory reset. The visual pose window remains the accepted ten rig clones, independently of gradient-history length. Full-session chronological validation checks long-term drift that short training gradients may miss. No claim is made that the proposed batch/chunk fits measured M5 limits yet.

## Loss and uncertainty treatment

Recommend **weighted, normalized MSE** of horizontal position, wrapped heading and reliable derived horizontal velocity as the first task loss, plus the agreed event-aware physical priors/regularization. Freeze unit/channel scales from training-only calibration/statistics and tune task weights on validation. Heading differences wrap across the angle boundary. Reference invalidity/quality masks must be defined independently of a method's own trajectory error.

Evaluate the loss at a declared common scoring cadence, initially 5 Hz, rather than weighting samples according to whichever sensor generated more events. Reference interpolation and velocity derivation are label processing, not online inputs. Validate coverage, interpolation and derived-velocity quality; retain graph-node/dense-reference sensitivity checks because NCLT reference information partly overlaps our sensor inputs. Do not treat dense differentiated pose as independent measured velocity.

Make uncertainty calibration/consistency diagnostics mandatory alongside task accuracy: pre-update innovations, coverage, covariance saturation, accepted/rejected observations and recovery. Initially train the covariance heads through the task loss and inspect these diagnostics. Introduce a small likelihood/calibration term once its covariance, current-observation conditioning and gating/reference-correlation assumptions are verified; its exact form/weight remains implementation work. This avoids claiming physically true covariance merely from an accuracy-trained multiplier. Preserve the fixed-process-noise first pilot ablation.

## Training stages and robustness

1. Establish a working, profiled analytical baseline with learned means zero and multipliers one.
2. Train mean heads with fixed baseline observation noise; isolate bias drift and wheel residual in ablations.
3. Train the observation-uncertainty heads, then jointly fine-tune the pilot. Include the corresponding uncertainty-only ablation under a matched budget.
4. Pursue the agreed full supported set subsequently, including process uncertainty, gains/FOG adaptation and the in-project lateral no-slip experiment. Exact expansion ordering remains its own decision.

Use modest training-only sensor corruptions/dropouts and degraded-vision examples to exercise adaptation/recovery. Label synthetic wheel corruption as synthetic corruption, not measured physical slip. Image blur/occlusion augmentation requires running the frozen frontend on the altered images; unchanged good matches are not a valid blurred-image experiment. Preserve realistic metadata/timing and positive noise models. IMU loss is not an ordinary skipped correction; use only supported fallback/failure behavior.

Relax the appropriate regularization during supported degradation and permit fast observation-weight/rejection changes; preserve the navigation loss on valid reference intervals. Do not turn failure periods into unscored training examples merely because the method struggles. Do not make arbitrary inconsistent edits to the analytical state/covariance correction.

## Validation, stopping and model selection

Use a fixed set of validation clips for rapid debugging, with periodic complete chronological replay of both validation recordings. Select final checkpoints on a predefined normalized navigation metric plus explicit failure/completion behavior; inspect uncertainty diagnostics rather than using the training loss alone. An estimator that scores only a short successful segment is not rewarded for hiding failures.

Start with early stopping after five full-validation checks without improvement, subject to a profiled compute budget. Hold method choices fixed before final test comparison; repeated final seeds/ablations are allowed, but test results must not drive further tuning presented as an untouched test. Network weights stay frozen at inference.

The next design group is evaluation/comparisons. Implementation/runtime feasibility should be checked alongside these decisions; deeper linearization/data-driven modelling stays last as requested.
