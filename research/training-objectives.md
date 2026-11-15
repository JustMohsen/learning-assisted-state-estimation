# Training objectives and literature precedents

14 November 2026. **Training-goal recommendations accepted, with event-aware regularization.** The accepted starting network has two causal GRU layers with 64 hidden units each, adjustable during development. Learned means initialize at zero, observation-variance multipliers at one; filter arithmetic is Float64 and the learned module Float32. Exact losses/weights, calibration likelihood and data protocol remain implementation/design details. Linearization/data-driven modelling remains the final dedicated design discussion.

## Accepted primary goal

Train the learned heads to improve the state estimates produced by the complete analytical ESKF/MSCKF recursion over sequences. Score multiple valid times, not only the final trajectory endpoint. The network outputs enter the filter as bias drift, wheel-model mean residual and measurement-variance multipliers; the Kalman gain remains analytical.

```text
causal sensors/context -> GRU + heads -> propagation/observation parameters
                       -> analytical filter propagation and corrections
                       -> estimated state sequence
                       -> reference-based loss
                       -> backpropagation through filter/time to neural weights
```

Accepted loss structure; exact terms and weights still to specify:

```text
loss = weighted navigation error over the sequence
       + small, event-aware mean-parameter regularization
       + an uncertainty-calibration term to specify and validate

navigation error includes:
  North/East position error
  wrapped heading error
  North/East velocity error where the derived reference is reliable
```

Normalize/weight different physical units explicitly. Exact MSE versus robust-loss choice, weights, sequence lengths and calibration term are still decisions for the training protocol. Use valid reference coverage at declared scoring times so sensor arrival density does not inadvertently change the objective. Velocity is derived from the NCLT pose reference, not an established independent velocity measurement; derivative quality, masking/smoothing and weights must be disclosed. Reported position/velocity use the agreed axle-center reference point.

The pose reference does not label true bias drift, sensor gains, slip or per-sample covariance. Those heads can be learned indirectly through navigation performance with priors/bounds; successful navigation does not prove unique recovery of the physical parameters. Synthetic perturbations or well-supported auxiliary labels can be used for pretraining/diagnostics, without presenting them as actual NCLT bias/covariance truth.

## Accepted caveat: allow rapid responses to real degradation

Use head-specific priors and regularization. A common smoothness or small-output penalty must not force the estimator to trust slipping wheels or degraded images.

| Quantity / event | Policy |
|---|---|
| Physical sensor biases and stable calibration | Retain physically justified bounded evolution and cautious adaptation; an observation failure is not evidence that IMU bias suddenly jumped |
| Wheel-model residual | Prefer modest/smooth corrections in healthy regimes, but relax its change penalty when independently supported wheel degradation makes a rapid mean correction useful. Hard output bounds remain, and severe/unpredictable slip uses rejection |
| Observation covariance / quality | Permit immediate increases in uncertainty or rejection on degradation; use little or no penalty on the required change. Return to normal trust with the agreed hysteresis/recovery checks |
| Visual blur, obstruction or dominant moving foreground | Use available quality, spatial coverage and geometric/inertial consistency; inflate uncertainty or reject unsupported observations. Complete absence of usable tracks means no visual update, rather than a fictitious huge-noise measurement |
| Process uncertainty in subsequent stages | Distinguish fast residual disturbances from slow calibration evolution; use appropriate channel-specific dynamics/regularization |

Regularization strength can depend on causal validity/quality evidence and pre-update consistency, with fixed-baseline safeguards and corroborating sensors where available. Do not let a freely learned head turn its own penalty off without constraint. Handle actual elapsed time and missing observations explicitly. Hard gates are not claims of perfectly labeled physical slip; exact thresholds and differentiability treatment are implementation details.

Relax the relevant **regularizer**, not the navigation-performance objective for otherwise valid reference intervals. Train/evaluate transients and recovery, including supported synthetic dropouts/blur/corruption, to detect a slow response induced by smoothness penalties. Preserve navigation/calibration covariance consistency; arbitrary clipping of the analytical correction or covariance is not the chosen recovery mechanism. No new mandatory event-classifier network is introduced.

## Accuracy and uncertainty are separate requirements

Accuracy-only training can learn useful effective observation weights whose values differ from physical sensor covariance. Evaluate calibration/consistency as well as trajectory error. Candidate secondary objectives include a properly constructed innovation likelihood or navigation-state likelihood; their covariance, reference uncertainty and correlation assumptions must be checked. A Gaussian likelihood includes both normalized squared error and a log-determinant term; merely minimizing an error divided by predicted variance encourages variance inflation. Finite bounds alone also do not establish calibration.

Distinguish adapting sensor noise R from learning the whole state covariance P. The accepted architecture predicts selected noise multipliers and propagates P analytically. Simultaneous Q/R/mean learning has identifiability tradeoffs; preserve the locked fixed-process-noise pilot and head ablations.

## What related studies train

| Study | Training target verified in primary sources | Relevance |
|---|---|---|
| Song et al., integrated-navigation KalmanNet, SPL 2024 | The inspected NCLT wheel/GPS configuration selects MSE on X/Y position. The official trainer compares masked recursive predictions with reference positions in windows and backpropagates with state/memory detachment | Direct NCLT precedent for supervising the resulting navigation estimates. It learns gain-related quantities, unlike our analytical-gain design. [Configuration](https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/configs/nclt/fusion/wheel_gpsfusion_origin.py), [trainer](https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/trainer/fusion_trainer.py) |
| AI-IMU Dead-Reckoning, 2019 preprint / T-IV 2020 | Trains an observation-noise adapter through an IEKF using relative translation error; the paper explicitly prioritizes localization accuracy rather than true covariance values | Older training precedent for noise adaptation; not a proposed replacement of our externally aided task. [IV-C](https://arxiv.org/html/1904.06064) |
| HDVIO, RSS 2023 | Residual thrust model trained on MSE of preintegrated position/velocity changes, without requiring force labels | Motion-supervised residual-model training, rather than backpropagating the final full VIO trajectory loss. [III-D](https://arxiv.org/html/2306.11429) |
| HDVIO2.0, T-RO 2025 | Residual thrust uses position/velocity-change MSE; residual torque uses orientation-change MSE. These models subsequently provide dynamics constraints in VIO | Shows that residual physical outputs can be trained from motion supervision without direct force labels. [Equations 7–8](https://arxiv.org/html/2504.00969) |
| Węgrzynowski et al., IROS 2024 racing study | Pretrains dynamics using one-step prediction, then fine-tunes dynamics/noise models through a differentiable UKF using sequence state-estimation MSE | Close training precedent for our proposed task-level objective with an analytical correction. [III-B4](https://arxiv.org/html/2408.15610) |

These are different training choices; hybrid estimators do not all optimize the final complete estimator end to end.

## Training staging and implementation implications

1. Reproduce the analytical baseline with learning disabled and validate its sensor models/noise.
2. Train the pilot heads separately through the filter, then together; preserve the agreed full-target scope.
3. Use pretraining only where a defensible auxiliary target exists or optimization needs it. No fabricated bias/covariance labels are required.
4. Validate complete held-out sequences, uncertainty behavior, failure/recovery and sensor dropouts; keep network weights fixed at inference.

Training requires a differentiable implementation of the selected numerical recursion; a stock C++ baseline is not automatically differentiable. Frozen visual models, matching, masks and discrete RANSAC/gating are not newly trained. Cache causal frontend results and specify treatment of state-dependent gates/track selection outside the gradient path. Truncated backpropagation limits gradient history, while contiguous filter/GRU state can persist as already agreed. Do not repeatedly replace the running estimate with reference state inside a sequence or provide test reference errors as neural inputs.

The current [training-protocol proposal](./training-protocol.md) supplies specific split dates, initial truncation/batching/optimizer settings, a first task loss, staged learning and validation policy. The protocol recommendations are accepted adjustable starting points; exact reference-quality treatment and calibration-loss construction remain implementation work. This note does not settle enhanced linearization or data-driven modelling choices reserved for the final design discussion.
