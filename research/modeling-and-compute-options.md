# Modeling, visual processing and training feasibility

Research checked: 13 November 2026. Hardware reported by the user: MacBook Air M5, 16 GB RAM. Application scope: indoor drones and road vehicles. No training throughput or baseline has been measured on this computer.

**Current dataset decision:** NCLT is the first development, training and evaluation dataset. VID indoor and the IROS 2024 racing-car data are optional future investigations. The [NCLT modeling note](./nclt-modeling-options.md) is the current discussion of parameters, nonlinearities, residuals and linearization. Novelty is not a project requirement.

**Scope update:** frozen **SuperPoint + LightGlue** is the initial visual route. Other learned/classical front ends and visual fine-tuning are optional investigations, not required work. Kalman gain/correction remain analytical. Training a visual model from scratch and building a new general VO/SLAM system are excluded. Earlier large-training examples below are context, not planned work. See the current [architecture discussion and flowcharts](./architecture-options.md).

## 1. Scientific objective

The objective is a better state estimator: improve accuracy relative to a relevant state-of-the-art system, or achieve comparable accuracy with a useful advantage in compute, latency, causal operation, retained history, or sensor requirements. Estimating forces or other auxiliary quantities is valuable only insofar as it supports that objective. NCLT, the initial camera route and analytical Kalman correction are agreed; a complete model/estimator architecture remains open.

The earlier EuRoC experiment was camera-free and effectively dead reckoning over the full trajectory. It should not be compared directly to camera-plus-IMU SLAM results. The user does not request debugging that experiment and remains open to EuRoC.

## 2. Camera processing: the proposed categories are broadly right, with additions

Camera processing involves more than selecting a feature detector. A representative geometric pipeline detects/tracks or matches image information, rejects inconsistent correspondences, applies camera/stereo geometry, and incorporates the observations into a filter or optimization backend. Calibration, timing, coordinate frames, failure handling and uncertainty are necessary regardless of whether the first component is learned.

The broader design landscape includes the following. Only classical/frozen processing is currently active; fine-tuning is optional/deferred, and from-scratch visual training is outside project scope.

- Hand-designed keypoints/descriptors and tracking: ORB, FAST/Shi–Tomasi features, KLT-style tracking, and related methods.
- Direct or semi-direct photometric methods, which use image intensity residuals and need not follow an ORB-style descriptor pipeline.
- Pretrained learned local features and matching used without further training, such as SuperPoint with LightGlue. LightGlue is a learned matcher rather than simply a CNN replacing a detector.
- Fine-tuning an existing detector, matcher or visual encoder on suitable training data.
- Training an existing compact architecture from random weights, versus inventing a competitive new visual architecture. These are different research burdens.
- More extensively learned systems that predict correspondence, depth or motion and may contain differentiable geometric optimization. A network predicting pose increments is still subject to observability and drift limitations.

Pretrained models do not always need adaptation. A frozen visual model combined with a small learned dynamics/uncertainty component is a coherent research design. It can keep the scientific contribution in state estimation.

Replacing ORB features in a mature system is not necessarily a direct substitution: descriptor representation, matching, feature management, vocabulary/place recognition and calibration assumptions can be coupled. Better pairwise matching does not by itself establish a better state estimator.

[SuperPoint paper](https://arxiv.org/abs/1712.07629), [LightGlue official implementation](https://github.com/cvg/LightGlue), [feature/matching training library](https://github.com/cvg/glue-factory), [ORB-SLAM3 implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3).

### Indicative engineering effort

These are planning judgments for one experienced researcher/developer with usable data and a working software environment. They are not published measurements, commitments or total project durations. Failures in calibration, platform support or training data can dominate the schedule.

| Choice | Current scope | Approximate effort to a useful integrated experiment | Training burden on M5 / 16 GB |
|---|---|---|---|
| Integrate an existing classical visual front end and estimator | Active option | Roughly 1–3 weeks | No visual training; small learned estimator modules are plausible locally |
| Integrate a frozen pretrained visual front end | Active option | Roughly 2–6 weeks | Visual inference and caching; local feasibility depends on implementation/operators/resolution |
| Fine-tune an existing compact visual model | Deferred and optional | Roughly 4–10 weeks including data/loss/evaluation work | Small jobs may fit; cloud makes repeated experiments more practical |
| Train a task-specific compact visual model from scratch | Excluded | Earlier estimate: roughly 2–6 months | Background context only |
| Develop a competitive general learned VO/SLAM system and training pipeline | Excluded | Earlier estimate: roughly 6–12+ months | Background context only |

Training an ordinary image-classification CNN is not sufficient preparation for a strong visual measurement model: correspondences, geometry, supervision and robustness have to suit state estimation. Short navigation datasets alone are weak foundations for training a general visual front end from scratch.

## 3. What is feasible on the MacBook Air?

PyTorch supports Apple GPU training through its MPS backend. The reported M5 is Apple silicon, but this note does not establish the installed framework/macOS environment or compatibility of a particular estimator repository. The 16 GB is shared system memory, not 16 GB reserved entirely for training.

**Reasonable local starting workloads:** low-dimensional residual MLPs, small recurrent models, covariance-scaling networks, numerical sensor fusion, and learning on cached visual observations/features. For small state dimensions a CPU implementation can also be practical. Small-batch, reduced-resolution visual fine-tuning may fit, but requires profiling before predicting run duration.

**More constrained workloads:** training long sequences through both a visual encoder and estimator, large dense correspondence/depth models, and pipelines with custom CUDA kernels. Existing NVIDIA/CUDA dependencies are a separate compatibility issue from theoretical model size.

MPS does not support float64 tensors. Numerically sensitive filter/optimization calculations may need CPU float64 or a carefully conditioned float32 formulation. Small-state filtering on CPU and neural training on a supported backend are possible implementation choices; they are not a universal acceleration guarantee.

If the camera model is frozen, its outputs can be cached once for appropriate estimator-training experiments. If that model is being fine-tuned, cached features from its old weights cannot replace its changing forward/backward pass.

[Apple MPS/PyTorch guidance](https://developer.apple.com/metal/pytorch/), [PyTorch MPS documentation and precision limitation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Published recipes illustrate the difference in scale

The official DROID-SLAM repository describes training with four RTX 3090 GPUs for about one week, or about 672 GPU-hours. Its implementation requires CUDA and specifies at least 24 GB GPU memory for training. This is an example of a full learned visual-SLAM training recipe, not a requirement for every learned visual feature or estimator network. [DROID-SLAM training documentation](https://github.com/princeton-vl/DROID-SLAM).

Glue Factory's published LightGlue recipe uses two 24 GB RTX 3090s for its default pretraining batch, with large image-training datasets; reduced batches are documented for weaker GPUs. A pretrained matcher can nevertheless be used without reproducing that training. [Glue Factory training documentation](https://github.com/cvg/glue-factory#training).

## 4. Cloud costs: published rates versus estimated budgets

Rates below were checked on official public pricing pages on 13 November 2026. Amounts are USD. Availability, cloud class and provider terms affect actual billing. GPU-hours are summed across GPUs.

| Example listed resource | Published rate | Scope |
|---|---|---|
| Runpod RTX 3090, 24 GB | $0.50/hour | Listed Pod rate; storage and other billed resources need separate accounting |
| Runpod RTX 4090, 24 GB | $0.89/hour | Listed Pod rate; select the actual offering before budgeting |
| Modal L4, 24 GB | $0.000222/second = $0.7992/hour | GPU component; CPU/memory and other charges are separate |
| Modal A100, 80 GB | $0.000694/second = $2.4984/hour | GPU component; CPU/memory and other charges are separate |

[Runpod official pricing](https://www.runpod.io/pricing), [Modal official pricing](https://modal.com/pricing).

The following are **illustrative compute allocations**, not measured training-time forecasts for an unspecified model:

| Allocation on one listed $0.89/hour RTX 4090 | Compute subtotal | Possible use, subject to profiling |
|---|---|---|
| 10 GPU-hours | $8.90 | A compact trial or feature extraction |
| 50 GPU-hours | $44.50 | Several short trials or a fine-tuning experiment |
| 200 GPU-hours | $178.00 | A limited repeated-experiment study |
| 500 GPU-hours | $445.00 | A larger training allocation; still no assurance of competitive results |

The cited DROID-SLAM recipe's 672 GPU-hours, priced arithmetically at the listed $0.50/3090-hour rate, would be about $336 in GPU-instance time for one recipe run. This is not a quote for a readily available four-GPU machine; hardware topology, storage, setup, repeated seeds, tuning and failed runs can change the bill substantially.

For this project, a local small estimator network or frozen visual front end could avoid extra cloud training cost initially. A few hundred dollars can fund meaningful compact-model experimentation, but cannot establish an upper bound on a new computer-vision research program. A short throughput/memory profile is needed before estimating epochs, wall-clock time or total experiment cost. Dataset sizes are not GPU-memory requirements: large recordings can be streamed.

## 5. Bias, physical residuals and covariance can be combined

A useful conceptual distinction is **systematic error versus uncertainty**. A bias or learned mean residual corrects a repeatable directional error. A covariance expresses the spread of uncertainty that remains under the model and available information.

A schematic formulation is

```text
state[t+1] = physics(state[t], inputs[t]) + mean_residual_network(eta[t])
             + G[t] * w[t]
w[t] follows Normal(mean=0, covariance=Q_network(eta[t]))
```

```text
measurement[s,t] = sensor_model_s(state[t]) + sensor_bias[s,t] + v[s,t]
v[s,t] follows Normal(mean=0, covariance=R_network_s(eta[t]))
```

Here eta denotes causal sensor/innovation/context history. A complete model might augment the state with sensor or actuator biases; the equations do not require learning all quantities simultaneously. Orientation needs an appropriate rotation representation, and temporally correlated disturbances require more than an independent white-noise assumption if that correlation matters.

| Example | Possible mean treatment | Possible uncertainty treatment |
|---|---|---|
| IMU bias/scale error | Bias state or calibrated/learned correction | Bias evolution and propagation-noise covariance |
| Motor command versus actual thrust | Physical actuator model plus gain/offset/residual correction | Actuator uncertainty mapped into state propagation |
| Unmodeled aerodynamic or tire effects | Learned residual or physical parameter adaptation | Remaining process uncertainty |
| Unpredictable external perturbations | Mean residual if necessary; explicit physical force estimate is optional | Effective disturbance/process covariance |
| Camera, UWB or GNSS observations | Appropriate measurement/bias model | Measurement covariance or robust likelihood |

Q and R are roles in a model, not intrinsic labels attached to hardware. For example, IMU noise contributes to propagation uncertainty when the IMU drives propagation; the same sensor can be an observation in an actuator-driven formulation. Posterior covariance P results from propagation and updates and must be evaluated separately.

### Can we learn wind uncertainty without estimating wind?

Yes: a learned effective process covariance can represent greater uncertainty during unmodeled disturbances. The network need not output wind speed, direction or force.

Two qualifications matter:

- Persistent directional disturbance can create a nonzero mean error. Covariance inflation may improve measurement weighting but does not by itself remove the cause. A learned mean residual or bias model may still help without claiming an explicit wind estimate.
- Without wind labels or an identifiable model, a learned Q is generally **effective process uncertainty**, not uniquely attributable wind covariance. It can also absorb actuator error, parameter mismatch, timing problems or sensor bias. Separating all those contributions from the same innovations is not generally identifiable.

Learning several full covariance matrices and many bias/residual components simultaneously can let modules compensate for each other. A practical study should start with calibrated nominal uncertainty, a small number of positive covariance scales or structured matrices, and clear ablations. Positive definiteness is necessary but does not prove calibration.

Relevant precedent includes [hybrid dynamics VIO](https://arxiv.org/abs/2306.11429), [learning vehicle models through a UKF with heteroscedastic noise](https://arxiv.org/html/2408.15610), and [A-KIT process covariance adaptation](https://arxiv.org/abs/2401.09987). These establish relevant design patterns rather than guaranteed benefits on our eventual task.

### Can actuator dynamics and IMU propagation coexist?

Yes, for example by using actuator-driven dynamics for prediction and the IMU as an observation, or using IMU propagation with additional uncertain dynamics constraints. The latter must account for information reused between the learned dynamics component, IMU propagation and updates; independent fusion of correlated copies can produce overconfidence. The exact architecture is still a research/design choice.

This supports the user's interest in combining physical-model correction with reliability adaptation. It does not require simultaneously estimating every actuator bias, every sensor bias and every environmental covariance.

## 6. UWB in UWB/IMU fusion

UWB means **ultra-wideband radio**. Known-position anchors communicate with a tag on the robot. Radio timing supplies ranges in two-way ranging (TWR), or range-difference information in time-difference-of-arrival (TDoA) systems. A filter fuses those observations with fast inertial propagation. UWB does not directly measure a full pose, velocity or all IMU biases.

The anchor system gives external spatial constraints in its coordinate frame; the IMU supplies high-rate motion information between radio observations. The approach requires installed/calibrated infrastructure and is affected by occlusion, multipath and geometry. The dataset's motion-capture reference can remain entirely separate from the UWB/IMU estimator inputs. Cameras are optional for such a fusion experiment.

[Bitcraze system explanation](https://www.bitcraze.io/products/loco-positioning-system/), [ranging-mode documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-loco-positioning-system/).

## 7. Indoor drones versus road vehicles: first focused comparison

These remain candidate application/dataset pairs, not selections. The common objective is state-estimation quality and useful deployment tradeoffs.

| Criterion | Indoor drones | Road vehicles |
|---|---|---|
| State and nominal physics | 3D position, velocity, attitude and biases; rigid-body/thrust/gravity dynamics where actuation data exist | Planar or 3D navigation plus velocity/yaw states; wheel, steering and single-track dynamics where channels exist |
| Useful model uncertainty | Thrust/actuator mismatch, aerodynamic effects, IMU errors, changing observation reliability | Tire/slip/steering effects, actuator/model mismatch, IMU/wheel errors, changing observation reliability |
| Observation choices | Camera/stereo, UWB, height/flow, IMU; exact combinations depend on dataset | IMU, wheel speeds, steering, GNSS, optional camera/LiDAR; exact combinations depend on dataset |
| Reference strengths | External mocap often provides clearer pose separation; force labels only in selected experiments | Optical velocity/sideslip instruments can validate local dynamic states; GNSS/INS reference may support trajectory claims |
| Practical entry | Classical or frozen visual front end; alternatively numerical UWB fusion; small learned estimator modules | Numerical wheel/IMU/steering fusion can avoid image training; full navigation still needs suitable spatial observations |
| Candidate dynamics data | VID; NeuroBEM as complementary model data | KIT Multi-Surface Driving Maneuvers; F1TENTH papers as relevant methods/data to audit |
| Candidate navigation data | EuRoC for VIO comparisons; UTIL/MILUV for UWB fusion | UrbanNav/Boreas for navigation; actuator/model channels must be verified |
| Main selection risk | Combining broad sensors, adequate independent training diversity and complete actuation/reference data in one release | Finding verified channels/reference and comparable published estimators for the exact state vector rather than a different vehicle task |

For navigation with drift control, velocity/sideslip estimation alone is not a substitute for a complete trajectory estimator. Conversely, a velocity/yaw state estimator is a legitimate physical-state estimation task if chosen explicitly. The user has not yet fixed the exact state vector.

The earlier paired-dataset comparison is historical: **NCLT is now selected first**. Continue with its precise sensor/model/reference contract. VID indoor and the 2024 racing-car data are optional future work. Visual fine-tuning remains deferred and optional; no from-scratch visual training or new general VO/SLAM system is planned.
