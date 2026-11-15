# Full-set rollout order — accepted

14 November 2026. **Rollout recommendations accepted by the user.** The full supported calibration/residual/uncertainty inventory is a committed research target. The sequence below orders its investigation; the pilot does not replace that scope. Numerical thresholds and exact later-stage heads can be refined during implementation. Current final discussion: [model mismatch, linearization/data-driven modelling and Gemma 4 feasibility](./model-learning-and-gemma4.md).

## Recommended sequence

| Stage | Investigate / implement | Reason for this order |
|---|---|---|
| 0. Analytical baseline | Validated IMU/FOG, wheel, processed GNSS and two-camera ESKF/MSCKF with the accepted frozen visual frontend/masks; fixed calibrated noise and ordinary analytical bias estimation | Establish functioning physics, observations, timing, covariance and compute before learned additions |
| 1. Locked pilot | Six IMU bias-drift outputs, one forward wheel-model residual, two GNSS variance multipliers, one wheel multiplier and one multiplier per accepted visual feature-observation block; general process noise/gains fixed, FOG bias analytical | Complete the agreed head-by-head and combined pilot under the accepted training/evaluation protocol |
| 2. Structured process uncertainty | Activate the process head: first selected IMU sample-noise intensities, then bias-evolution uncertainty, then justified residual model/environment process channels. Introduce these families separately before combining | Extends the already defined physical propagation without introducing more calibration means at the same time |
| 3. Lateral wheel constraint | Implement/evaluate a soft lateral no-slip relation and its reliability/variance treatment; compare on/off and inflate/reject during supported lateral sliding | Committed subsequent work within this project; checks whether another physical constraint adds useful information while keeping wheel scale fixed |
| 4. FOG adaptation | Extend the existing filter-owned FOG rate bias with learned drift and interval-noise/reliability adaptation in the joint angular subsystem; initially keep its gain fixed | Reuses existing fusion and diagnostics. Preserve interval correlations, IMU fallback and joint calibration uncertainty; no second independent heading update from the same FOG data |
| 5. Slow gain/calibration adaptation | Common wheel scale first, then supported gyro/FOG gains and accelerometer gains, axis/family at a time. Compare fixed train-calibrated values, analytical slow-state adaptation and learned assistance | Scale, bias and mean residuals can explain overlapping errors. Establish excitation/observability and constrain competing corrections before joint adaptation |
| 6. Richer uncertainty and full-set combination | Investigate full GNSS 2x2 covariance, anisotropic visual 2x2 blocks and justified inertial/process blocks; assess the full supported combination against isolated contributions | Tests remaining structure after individual roles are understandable. Keep covariance positive and transformed consistently; retain the analytical state covariance and gain |

This is a research order, not a requirement to finish every experiment in one stage before any independent work in another can begin. Process-noise adaptation follows the pilot as previously planned; the lateral constraint is explicitly in-project work rather than optional future work.

## Identification and model ownership

Introduce one parameter/noise family at a time. For wheel-scale identification, hold the flexible wheel residual fixed or constrain it sufficiently to avoid fitting the same scale discrepancy twice; use reliable external motion information. Review IMU/FOG relative versus absolute bias/gain identifiability using their mounting geometry and available external observations.

Investigate the proposed gain families, but activate online estimation only for quantities supported by the recording's excitation and observations. Retain nominal/offline priors for weakly observable axes, and document the limitation rather than interpreting arbitrary neural outputs as true physical calibration. Begin with the simpler train-calibrated/analytical-state alternatives so a per-event neural gain has a fair comparator.

Sensor sample noise, bias-evolution noise and residual process mismatch have distinct injection points. Avoid enabling multiple unconstrained noise channels to explain the same error. Environmental uncertainty means residual prediction error under the chosen model; acceleration from external forces already measured by the IMU is not a separate independent force/noise input to add again.

The full inventory does not imply an unrestricted neural replacement for the physical transition, P or Kalman gain. Published/verified camera intrinsics, mounting and time calibration remain the accepted initial geometry. Additional residual/submodels, model parameters, actuator/mass/inertia ideas and enhanced linearization are reserved for the final discussion; suitability depends on the measurements actually available. Unverified actuator command/torque channels are not assumed in NCLT. VID/racing remain optional other-dataset investigations.

## What advances into the combined estimator

- Check mathematical/numerical correctness, causality and supported observability before interpreting a new head or calibration state.
- Compare each family to the fixed baseline and simpler alternatives using validation, uncertainty/robustness behavior and measured compute, then evaluate the declared frozen methods on the held-out test protocol.
- Enable a component in the recommended combined configuration when the evidence supports its accuracy or practical benefit. Retain unfavorable/negative results; a family with no gain need not block other planned investigations.
- Pursue the full supported set as agreed. Numerical stability and parameter support are engineering/scientific conditions, not additional user-permission gates.

The remaining planned physics discussion is **linearization and data-driven modelling**, accompanied by the user's new question about Gemma 4. The [discussion note](./model-learning-and-gemma4.md) records recommendations without implicitly changing the locked dataset/backend/pilot scope or adding a Gemma dependency.
