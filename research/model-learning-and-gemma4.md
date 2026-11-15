# Model learning, linearization and Gemma 4 — discussion

14 November 2026. The [full-set rollout order](./full-set-rollout.md) and baseline grey-box design remain accepted. The user accepts the **post-filter neural refiner as a potential experiment/future work after the pilot**, and **deferred Gemma experiments: offline annotation/teacher first, then low-frequency supplemental scene context**. Neither changes the locked initial estimator. No Gemma model has been downloaded, fine-tuned or run. The consolidated design is in the [project snapshot](../archive/PROJECT_SNAPSHOT.md).

## Brief modelling explanation

Data-driven modelling learns a transition, observation relation or its parameters from data; retaining known physics and learning a structured residual is hybrid/grey-box modelling. Our bias evolution, wheel residual and noise heads already follow this pattern. A learned mean residual addresses systematic model mismatch; uncertainty adaptation describes remaining unpredictable error. State-reference supervision can train useful corrections without uniquely identifying the underlying physical parameters. [Relevant hybrid dynamics/filter example](https://arxiv.org/html/2408.15610).

Linearization error is different: the EKF already propagates a nonlinear nominal state, while its uncertainty/update approximation uses local first-order information. Learning a nonlinear residual and differentiating the corrected model gives consistent derivatives but does not remove omitted higher-order uncertainty effects. Keep the accepted observability-aware/FEJ policy and consistent Jacobians first. Directly learning an arbitrary model-matrix correction is not the recommended first extension; targeted moment/linearization correction would need a separate, controlled experiment. [FEJ and linearization explanation](https://docs.openvins.com/fej.html).

For this project, recommend retaining the bounded, interpretable residual/parameter channels and identifying any additional missing relation before adding a broader neural transition model. Mass/inertia or actuator dynamics require suitable independent force/torque/control information and excitation; these have not been established for NCLT. A neural correction's numerical usefulness is not proof that it estimates a unique physical force or parameter.

## Clarification: where nonlinearity already enters

The nominal state is propagated through nonlinear inertial/rotation equations, and predicted measurements use nonlinear geometry where applicable. The accepted GRU/MLP heads are nonlinear mappings of causal sensor/filter context. Their outputs affect means, parameters and uncertainty even though the core gain is analytical. F and H describe local error sensitivities; they do not replace all nominal propagation/measurement evaluation by linear functions.

```text
state_prediction       = nonlinear_physics(state, sensors, learned_parameters)
measurement_prediction = nonlinear_sensor_model(state_prediction, learned_parameters)
# Local Jacobians are used for covariance, gain and the error-state update.
```

This distinction follows the [inertial mean/error propagation](https://docs.openvins.com/propagation.html) and [measurement-update derivations](https://docs.openvins.com/update.html). Covariance/first-order approximation limitations still remain.

## Potential post-filter residual experiment — agreed future work

The user agrees to record this as an optional post-pilot investigation: a **causal residual readout/error refiner** estimates remaining navigation error after the analytical filter. It may compensate for systematic model mismatch or filtering approximations, but reference-error supervision cannot uniquely label the output as the unmodelled nonlinear part or isolate linearization error.

```text
context = local-motion/filter features + current GRU summary
          + recent innovations + covariance/quality + availability/age
delta_navigation = bounded_small_MLP(context)
refined_navigation = filter_navigation + delta_navigation
```

Recorded initial experiment direction after a functioning pilot; precise design remains adjustable:

- Freeze the trained core estimator/encoder initially; use a small MLP, for example two 32-unit hidden layers, with zero-initialized bounded correction outputs. Layer sizes and physical bounds remain tunable.
- Predict five reported-output corrections: horizontal position (2), horizontal velocity (2), heading (1). Use wrapped angle residuals/composition. Check temporal/kinematic consistency as well as pointwise error.
- Provide informative causal context rather than only the final position/heading. Identical estimated states can have different remaining errors. Prefer local-motion/relative features and check route memorization if absolute coordinates enter.
- Train on reference-minus-filter residuals within the training partition. Use validation for selection; consider held-out/out-of-fold core predictions within training to avoid optimistic in-sample residual fitting. Do not tune on test errors.
- Keep raw filter output and refined output separately, and initially propagate the next step from the original filter state. The refiner is not an independent measurement to fuse a second time.
- Evaluate identical sensor/history/compute conditions, including faults and recovery. Compare the frozen-pilot baseline to baseline-plus-refiner; investigate joint training or recursive feedback only after a useful result and a consistent uncertainty model.

The original EKF covariance is not automatically calibrated for the refined output. A suitable residual uncertainty/error model must account for correlations and uncertain context; a naive reuse of P or an assumed-independent extra sensor is insufficient. This adds an experimental output branch, not an additional Kalman gain learner. It does not require abandoning the committed rollout.

A related error-compensation example is [LSTM-based Kalman Filter Error Compensation for Enhanced Moving Object Tracking, 2025](https://doi.org/10.5302/J.ICROS.2025.24.0143). It concerns constant-velocity object tracking, not evidence of performance for our NCLT multi-sensor architecture.

## What Gemma 4 can contribute

Google documents Gemma 4 as a multimodal text-output model with image/video-frame understanding, object detection/pointing and multiple-image reasoning. Those capabilities suggest optional semantic assistance, not a validated replacement for our calibrated keypoint/matching interface. [Vision documentation](https://ai.google.dev/gemma/docs/capabilities/vision), [model card](https://ai.google.dev/gemma/docs/core/model_card_4).

| Possible role | Assessment for our project |
|---|---|
| Replace SuperPoint/LightGlue | Poor first fit: our MSCKF consumes repeatable geometric tracks with calibrated coordinates. General image understanding/grounding does not establish equivalent descriptor, temporal-match or stereo performance |
| Assist dynamic-region masking | Potentially useful for unusual objects/context; any region proposals must be spatially validated. Text/boxes do not automatically supply accurate pixel masks. Keep the accepted Mask R-CNN baseline for comparison |
| Low-rate semantic/context input | Plausible experiment: potential visual obstruction, reflective scenes, unusual foreground or possible low-traction conditions could supply context to existing quality heads. These are hypotheses/cues, not measured slip, calibrated covariance or ground truth |
| Offline annotation/teacher | Strong initial candidate: annotate difficult training clips or provide auxiliary supervision to the already planned quality heads. Treat outputs as noisy pseudo-labels and review a sample; no new general visual-model training is required |

The user accepts these as **deferred future experiments**, in the following order; Gemma is not a required pilot dependency:

1. **Offline annotation/teacher during training.** Start with frozen quantized E2B on a small training-only set of difficult clips. Use reviewed, noisy auxiliary labels for the already planned quality heads, whose inference inputs must be deployable without the teacher. Compare annotation quality and resulting held-out navigation/recovery; do not treat generated labels as physical truth. No new general visual-model training or Gemma fine-tuning is required.
2. **Low-frequency supplemental scene context.** If useful, test sparse causal image/history calls that supply context to the existing quality subsystem. Use explicit timestamp/age/availability and continue estimation when Gemma is delayed or absent. Include inference cost and latency in the comparison; do not substitute unvalidated generated confidence for calibrated sensor covariance.

E4B is a quality alternative if E2B is inadequate. The selected SuperPoint/LightGlue/masking baseline remains in place. Cached future-frame scene analysis cannot become an online input; any teacher supervision and inference-time features must be distinguished.

## Local feasibility and cost

Google's approximate Q4_0 load-memory estimates are:

| Variant | Listed memory |
|---|---|
| E2B | 2.9 GB |
| E4B | 4.5 GB |
| 12B | 6.7 GB |
| 26B A4B | 14.4 GB |
| 31B | 17.5 GB |

These estimates do not include full runtime/context-cache costs and are not measured Mac footprints. On a 16-GB shared-memory Mac, E2B/E4B are plausible small-trial candidates; larger models need separate memory profiling, and 26B/31B leave inadequate room for the full pipeline under these estimates. Official QAT GGUF variants support Apple Silicon runtimes. [Google model overview and memory guidance](https://ai.google.dev/gemma/docs/core).

Local downloaded-weight inference has no API token bill, but consumes memory, compute and energy. Frozen use adds no model-training job. Do not promise native two-camera/5-Hz throughput: that is ten images per second alongside segmentation, matching and filtering, which has not been benchmarked. Short context, sparse calls and small structured outputs are a reasonable first experiment; full fine-tuning is not proposed.

## Hosted options and illustrative costs

As checked on 14 November 2026, Google's Gemini Developer API documentation lists Gemma 4 image input and a free tier, with no paid tier in that pricing entry. This is separate from the paid Google Cloud managed model; quotas, endpoint availability and actual image behavior require verification for a proposed run. [Hosted model/image-input documentation](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), [Developer API pricing](https://aistudio.google.com/docs/pricing#gemma-4).

Google Cloud lists Gemma 4 26B at **USD 0.15 per million input tokens and USD 0.60 per million output tokens**. Its managed model supports text/image input. [Cloud pricing](https://cloud.google.com/model-garden/pricing), [hosted model capabilities](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it).

Illustration only: assuming 1000 total image-plus-text input tokens and 50 output tokens per image gives **USD 0.18 per 1000 images**, or **USD 6.48 per hour** at two cameras and five images/second/camera. Actual image-token budgets, prompt/history, generated reasoning/output, quotas and service options change the bill and throughput. Frozen-output caching makes this a cache-generation cost rather than something to repeat each estimator-training epoch. These are not measured project costs.

Accepted future-experiment plan: retain the selected geometric frontend and small estimator heads; investigate offline Gemma annotation/teacher first, then low-frequency supplemental context. Exact future sampling/runtime choices refine during those experiments. Neither future experiment requires mandatory language-model inference in the initial estimator.
