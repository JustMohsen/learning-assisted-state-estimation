# Evaluation, comparisons, implementation and compute — accepted

14 November 2026. **Recommendations accepted by the user.** Evaluation/comparison policy, Python/PyTorch as the first prototype, reference reuse, incremental integration and local-first profiling are locked as adjustable engineering choices. OpenVINS and NCLT KalmanNet are the first external reference candidates; compatible adapters/protocols still require verification. Exact benchmark details and numerical targets can be refined while building. No estimator, package environment or runtime profile has been established. Rollout order is accepted. Current discussion: [model learning and Gemma 4](./model-learning-and-gemma4.md).

## Evaluation and comparisons

| Decision | Recommendation |
|---|---|
| Primary accuracy | Horizontal position RMSE, wrapped-heading RMSE and quality-controlled derived horizontal-velocity RMSE, reported by test day and across the held-out set |
| Trajectory/drift | Add relative pose/drift errors on declared intervals and GNSS-outage segments. Standard aligned ATE/RPE can be supplementary when required for a published comparison |
| Reference and alignment | Use validated graph-node pose scoring where available and dense-reference sensitivity checks. Keep primary global errors in the declared navigation frame/initialization, without fitting the whole test trajectory. Label supplemental alignment explicitly; no scale fitting for metric outputs |
| Primary controlled comparator | The identical analytical ESKF/MSCKF with neural means zero and noise multipliers one, retaining ordinary analytical bias estimation, the same sensor inputs, frozen visual frontend, masks, history, initialization and gates |
| External classical reference | OpenVINS is the first candidate for a compatible camera/IMU subset, using its traditional tracker; adapt data/calibration and validate compatibility before committing a reproduction. Stock software is not assumed to support our full FOG/wheel/GNSS/custom-frontend system |
| External learned reference | Song et al.'s NCLT KalmanNet implementation is the first candidate. Match or explicitly separate its sensor budget, preprocessing, 1-Hz protocol, heading inputs and startup from our native-rate multimodal experiment |
| Learning ablations | Fixed baseline; bias-drift only; wheel-residual only; observation-uncertainty only; combined pilot. Pursue later-stage additions separately and together where useful |
| Robustness and consistency | Natural and controlled sensor loss/corruption, wheel unreliability, blur/occlusion/dynamic scenes and recovery. Record failures, resets, completion, innovation/coverage diagnostics and variance saturation. NEES is a diagnostic only where reference/covariance assumptions support its interpretation |
| Compute and deployment tradeoff | Report end-to-end latency including frozen vision, median/tail runtime, peak memory, processed sensor/camera budget and retained history. Cached-frontend replay timings are separate from actual full-system timings |
| Scientific success | Better held-out estimation, or comparable accuracy with a measured compute/history/latency/sensor advantage. Beating the internal baseline alone does not establish broad state-of-the-art superiority |

The fixed analytical-backend baseline still contains frozen SuperPoint/LightGlue and Mask R-CNN; call it a fixed analytical estimator, not an entirely classical visual pipeline. A traditional-tracker external reference supplies a distinct comparison without requiring the optional internal frontend sweep. Extra sensors, masks, loop closure or longer history must not be credited to learned calibration/residual heads.

Keep all failed/lost intervals in completion/failure accounting and report recovery; accuracy on valid estimated samples is not a substitute for completing the recording. Preserve the agreed initialization/reset policy. The reference shares some source information with our estimator, and derived velocity is not independent instrument truth; uncertainty/physical-calibration claims need that qualification. Exact tolerances and reference-quality rules are implementation/evaluation details, fixed before final test evaluation.

Relevant primary implementation sources: [OpenVINS feature list and evaluation support](https://github.com/rpng/open_vins), [NCLT learned-filter reference](https://github.com/SongJgit/KalmanNet4SensorFusion), [evo APE/RPE and alignment documentation](https://github.com/MichaelGrupp/evo/wiki/Metrics). Use a common evaluation adapter and cross-check standard trajectory metrics with an established tool; custom planar/velocity metrics remain explicit.

## Implementation and compute

| Decision | Recommendation |
|---|---|
| First implementation | Python/PyTorch research prototype with a differentiable analytical filter core. Use the same core with learned heads disabled/enabled for training and inference replay |
| Reuse | Frozen official visual models and existing utilities; OpenVINS derivations/reference behavior guide the MSCKF implementation. Reuse/adapt compatible pieces with consistent quaternion/error conventions rather than copying incompatible Jacobians |
| Integration order | Native numerical sensor/timestamp pipeline and analytical aided estimator; two-camera frontend/MSCKF integration; gradient/numerical verification; pilot learning. Early numerical-only runs are bring-up, not a reduction of the full visual pilot scope |
| Hardware | Start on the M5 MacBook Air/16 GB. Keep the Float64 filter on CPU. Evaluate Float32 neural/vision workloads on CPU or MPS where operations are supported and measured performance helps; retain a tested CPU path |
| Data and caches | Start with short recordings/samples and grow incrementally. Cache frozen masks/features/matches with configuration/weight/calibration provenance. Measure storage before staging the complete image corpus; selected-camera extraction may still require transferring a combined compressed archive |
| Profiling before expansion | Measure representative sustained runtime/memory and gradient-chunk cost. Use those results to tune batch size, camera count, image/track budget and gradient history while preserving scientific comparisons |
| Later acceleration | Port demonstrated bottlenecks to optimized code if needed, retaining numerical/trajectory agreement with the training core. Avoid committing to a second full implementation before profiling |
| Cloud compute | Optional after a measured local bottleneck justifies it; choose scale and budget from the profiling results |
| Reproducibility | Record dependency/source versions, model weights, split/configuration, calibration, data/cache provenance, seeds and evaluation settings |

Python/PyTorch is the accepted first codebase choice. A stock differentiable MSCKF supporting our complete workload is not assumed available. Verify frame/unit/timing models, error injection/reset, Jacobians, covariance behavior and training gradients before large runs. Implement the agreed event-driven clocks and causal delayed-observation handling. Existing C++ reference code is not automatically differentiable; a later accelerator needs a deliberate gradient/equivalence interface.

[PyTorch MPS documentation](https://docs.pytorch.org/docs/2.15/notes/mps.html) describes the Apple GPU backend. Actual support and throughput of our chosen operators, frozen segmentation/matching and mixed-device training still require a small feasibility run. No measured real-time or local-training-duration promise is made.

The current [rollout-order proposal](./full-set-rollout.md) covers the already committed full supported scope. The final linearization/data-driven-modelling discussion follows it, as requested by the user. The accepted evaluation/implementation choices do not settle that final topic.
