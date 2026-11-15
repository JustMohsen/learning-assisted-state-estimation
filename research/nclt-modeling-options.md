# NCLT modeling options and remaining decisions

Updated: 14 November 2026. Phase: discussion/design. **NCLT is locked as the first dataset for development, training and evaluation.** VID indoor and the Węgrzynowski et al. IROS 2024 racing-car data are optional future investigations. Novelty, publication and finding an unclaimed architecture are not project requirements. Follow a promising approach and assess it through useful state-estimation experiments.

The agreed task is **full planar navigation output: north/east position, north/east velocity and heading**, with tilt-aware 3D internal attitude. **IMU/FOG-assisted inertial propagation with wheel/GNSS/visual updates** and analytical gain/correction are agreed. Sensor adaptation, model residual and uncertainty adaptation are the three learning roles. Pilot ownership/outputs, input groups/clocks/memory and startup/recovery are locked; detailed fusion/noise mathematics and later-stage parameterization remain open. Magnetometer is disabled, potentially indefinitely; barometer is excluded. The camera design supports one through six views with all-pairs/reduced-pair modes. See the [sensor and state specification](./nclt-sensors-and-state.md). Use plain-text equations.

## Current items: model learning/linearization and Gemma 4

The user has **locked the backend, shared encoder/head organization, wheel-slip handling, exact pilot specification, input/timing/memory, startup/recovery, baseline mathematics, initial numerical/visual settings and training goals with event-aware regularization**. The full supported inventory remains the planned research target. [Training protocol](./training-protocol.md) is accepted as an adjustable starting point. [Evaluation/comparisons and Python/PyTorch/local-first implementation](./evaluation-and-implementation.md) are accepted. [Full-set rollout order](./full-set-rollout.md) is accepted. [Model learning/linearization and Gemma 4 feasibility](./model-learning-and-gemma4.md) are the current brief discussion. Begin with two cameras and verify the pair by usable overlap/triangulation rather than center distance alone. The starting GRU is two layers of 64 units, with adjustable configuration. Lateral no-slip investigation is subsequent work within this project; deeper linearization/data-driven modelling is the final design discussion. See the [remaining-decision checklist](./project-decisions.md#remaining-decisions-in-order).

### Locked backend and visual coupling

**User-accepted decision: error-state EKF with tightly coupled, MSCKF-style visual feature-track updates, a bounded history of cloned rig poses, and no persistent landmark map or loop closure in the first version.** Calibration ownership is now active. OpenVINS remains a reference/candidate baseline, not a selected implementation language/codebase.

These terms describe compatible parts of one estimator:

- **Error-state EKF:** propagate a nonlinear nominal navigation state, represent small perturbations in its covariance, analytically update those perturbations, inject them into the nominal state and reset the error representation/covariance consistently. It governs propagation uncertainty as well as correction. Error-state coordinates alone do not eliminate linearization error or guarantee consistency. [Solà's derivation](https://arxiv.org/abs/1711.02508).
- **Tightly coupled visual update:** masked, matched, geometrically checked pixel/bearing tracks enter the joint estimator directly. The main visual interface does not first compress them into an independently produced VO pose. Feature observations can correct navigation and supported calibration quantities through the shared state and cross-covariances.
- **MSCKF-style treatment:** keep copies of recent rig poses with their cross-covariances, triangulate temporary landmarks from tracks, form multi-view residuals and project out feature-position uncertainty. Do not keep every landmark permanently. [OpenVINS paper](https://pgeneva.com/downloads/papers/Geneva2020ICRA.pdf), [nullspace derivation](https://docs.openvins.com/update-null.html).

Proposed visual data path:

```text
Images -> frozen instance masks + SuperPoint features
       -> mask-filtered keypoints -> LightGlue matches -> persistent track IDs
       -> RANSAC and uncertainty-aware inertial/temporal consistency checks
       -> temporary landmark geometry + feature-track residuals
       -> robust analytical MSCKF-style update of the joint navigation estimator
```

IMU/FOG supplies propagation; wheel and processed GNSS observations have their own updates to that same state. Masking and robust geometry do not replace estimator covariance. A pose estimated for RANSAC verification is not also fused as a second independent observation derived from the same tracks. Repeated updates of active tracks must not reuse already-consumed measurement information without accounting for it; track completion/marginalization policy is part of implementation.

| Alternative | What is attractive | Why it is not the recommended first design |
|---|---|---|
| Separate VO pose followed by EKF fusion | Simpler interfaces when integrating an existing VO package | Derived-pose uncertainty, temporal correlations and any shared inertial priors require care; less direct control of how individual features constrain calibration |
| UKF with visual history | Sigma-point nonlinear uncertainty propagation | Rotation handling and many augmented pose/calibration dimensions complicate repeated transforms; it does not solve correspondence/landmark nuisance handling or dynamic-scene robustness automatically. No accuracy disadvantage is assumed |
| Sliding-window nonlinear optimization | Can relinearize recent states and observations, with mature visual-inertial examples | Solver convergence, marginalization and variable iteration costs add design choices; valuable if filter linearization limits performance, but not a requirement for the initial recursive-estimator experiment |

The tradeoff is project-specific, not a claim of universal EKF superiority. [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) is a primary optimization-based alternative. A finite-window optimizer can also operate causally; history length and processing delay are separate from use of future measurements.

Implementation implications and limits:

- Use the accepted adjustable starting point of ten rig-pose clones, up to 1024 retained background features per image and up to 200 accepted tracks per rig update. Measure usable parallax and latency at NCLT's 5-Hz imagery. Emit the current propagated state without waiting for a fixed number of future frames. Visual corrections still incur processing and track-maturation latency.
- For synchronized cameras, share one body/rig pose clone per timestamp and use individual calibrated camera transforms. Multiple camera observations increase constraint rows/feature work; they need not create six unrelated navigation states. Keep shared track identities and uncertainty coherent across simultaneous/temporal observations.
- The reported task remains five planar quantities. **Full 3D internal position/velocity/attitude is now accepted**, including vertical nuisance states for camera/rig geometry and slopes. This does not change the planar evaluation target. Planar constraints require physically justified models and noise.
- Use an observability-aware linearization policy, such as the FEJ treatment documented in OpenVINS, adapted to our external GNSS/wheel/FOG observations. The unobservable directions of unaided VIO are not blindly imposed when external measurements supply those constraints. [FEJ documentation](https://docs.openvins.com/fej.html).
- Adopt DynaVINS-style inertial checks and DynaVINS++ calibration-consistency principles, not their optimization backend unchanged. Robust filter updates/recovery must be derived consistently for our filter.
- **OpenVINS is a mathematical/implementation reference and a candidate baseline**, not yet the selected codebase or language. Its stock frontend and supported camera paths are not assumed to implement our six-camera LightGlue/masking/FOG system. Differentiable training through the recursion and integration of all new sensors are separate engineering work; C++ reference code is not automatically differentiable from a neural training framework.
- The backend is a plausible fit for bounded online computation; no measured M5 throughput exists. Frozen segmentation/matching may dominate total cost, so profile the complete system.

The [ownership inventory](./calibration-ownership.md) now lists quantity, frame, initialization/evolution, responsible component and injection point. Network architecture, exact heads and initial activation subset remain to be discussed.

| Component | Agreed basis / proposed detail | Decision still needed |
|---|---|---|
| Navigation outputs | **Agreed:** horizontal position/velocity and heading, five reported quantities | Numerical accuracy/latency targets; do not reopen planar versus 3D output |
| Internal attitude | **Agreed:** tilt-aware 3D attitude | Representation, gravity handling and explicit treatment of any vertical nuisance quantities needed by visual geometry |
| Sensor error quantities | **Categories agreed:** IMU biases in released sensor axes; selected scales; FOG bias/scale; wheel calibration where justified | Which are fixed/offline, augmented states, learned priors or slowly adapted quantities; avoid redundant compensation |
| Inertial propagation | **Agreed:** corrected IMU specific force plus analytically combined IMU/FOG angular information | Common causal intervals, mounting geometry, calibration/noise cross-correlations and dropout fallback |
| Wheel observation | Forward speed initially; soft lateral/no-slip constraint committed for subsequent investigation within this project | Detailed lateral uncertainty/slip handling, numerical limits and comparisons |
| GNSS observation | Receiver-processed horizontal antenna position | Position/frame/lever-arm model, gates and covariance |
| Visual observation | **Locked:** tightly coupled feature-track constraints with temporary 3D geometry | Exact track/history policy, geometry and correlations |
| Analytical backend | **Locked:** error-state EKF with MSCKF-style feature updates | Codebase/language, numerical window and detailed state/noise equations remain open |

The following three logical roles are **agreed**. Distinguish their exact outputs by **where they enter the model**:

| Logical neural role | Possible outputs | Necessary separation |
|---|---|---|
| Sensor calibration/error adaptation | Selected sensor-axis bias/scale corrections, bias evolution or parameter priors | One coherent sensor calibration model; no duplicate unrestricted filter/network bias |
| Model mismatch | A structured correction to one propagation or observation relation, e.g. wheel slip/no-slip mismatch | Defined units/frame/time behavior; cannot claim uniquely identified physical slip if calibration or timing can explain the same correction |
| Uncertainty/reliability | Structured process/sensor noise parameters or robust observation weights | Mean correction and uncertainty have different roles; retain positive covariances and a fixed/tuned-noise baseline |

**Locked organization:** one small shared GRU on IMU events with separate sensor-adaptation, model-residual and process-uncertainty heads, plus observation-triggered quality heads for GNSS, wheels and vision. Persistent causal memory and sequence-mini-batch training are agreed. Hidden size and numerical interfaces remain open. Evaluate modules individually and together; the full supported set is explicitly planned beyond the pilot.

**Dynamic visual data is an explicit error source.** A correct match on a moving object can be inconsistent with the stationary-world observation model. The user now selects **frozen DynaSLAM-style instance masks plus RANSAC, IMU/FOG-prior motion checks and calibration protection** for the baseline; see [architecture options](./architecture-options.md). Mask R-CNN is the first reference, with exact checkpoint/runtime and cadence still open. [DynaVINS++](https://arxiv.org/html/2410.15373v1) documents contamination of IMU bias estimates by bad visual associations. Gate untrusted visual information before physical calibration/residual adaptation, and evaluate calibration transients as well as pose error. Exact recovery/weighting must respect the backend's covariance treatment. Frozen segmentation adds no new trainable learning role or object-tracking/SLAM requirement.

The backend is settled. The order is now: calibration-state ownership and exact physical/error equations, selected neural input/output/cadence contracts, then training and evaluation protocol. The remainder supplies the physical/data-driven modeling background.

## Does this require mapping or SLAM?

A persistent global map is not required for GNSS/wheel/IMU/FOG-aided navigation. The closest learned-filter experiments reviewed here are numerical navigation tasks: [original KalmanNet, IV-E](https://arxiv.org/html/2107.10043) uses odometry velocities; [KalmanFormer, Section 4.3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1460255/full) uses IMU/consumer GPS. These NCLT experiments do not build maps or perform visual loop closure. Broader NCLT work does include mapping/SLAM, for example [LTA-OM, 2024](https://onlinelibrary.wiley.com/doi/10.1002/rob.22337), which evaluates LiDAR/IMU mapping and long-term association on NCLT. The dataset's reference also uses offline SLAM; that does not require our online estimator to do the same.

Distinguish three levels:

| Level | What is retained | Implication |
|---|---|---|
| Temporary visual geometry | Tracks and triangulated 3D points over a bounded recent interval | Supports motion constraints without a persistent global map |
| Local landmark mapping | Selected nearby landmarks and recent poses | Can improve tracking/constraints but adds state and marginalization management |
| Persistent global SLAM/relocalization | Long-lived map, place recognition, loop closure and potentially revised past poses | Additional information/history and substantial scope; comparisons must match its use |

**Locked:** bounded local visual tracks/pose history initially, with no persistent global map or loop closure. The selected MSCKF-style update forms constraints across retained poses while eliminating feature-state dependence. [OpenVINS overview](https://docs.openvins.com/), [primary nullspace derivation](https://docs.openvins.com/update-null.html). It is an architectural reference, not a claim that its implementation already supports our full six-camera/frozen-matcher/FOG configuration.

A bounded window of past poses can produce a causal estimate of the current state. Window length, processing delay and retroactive changes to earlier poses must be reported separately. Five planar outputs do not imply a flat scene: camera projection and temporary landmarks still use 3D geometry. Any internal height/vertical-motion treatment must be specified, without changing the agreed output task.

## Sensor parameters: initialize, retain, predict and correct

The user asks whether sensor errors are initialized and then re-estimated by the network every step. That is a possible design, not an automatic consequence of agreeing to sensor adaptation. Distinguish an online parameter estimate from online training of the network weights.

Recommended first treatment:

1. Initialize supported sensor-error quantities from nominal/offline calibration or a declared sensor-only initialization procedure, with uncertainty. A single stationary orientation cannot identify every accelerometer gain/bias independently.
2. Retain estimates across time. Bias can drift; a stable gain should not be freely replaced at every sample. Between adaptations, propagate or hold each quantity according to its model.
3. For bias, a candidate neural role is a bounded drift model inside augmented-state propagation:

```text
predicted_bias = previous_bias + bias_drift_network(causal_history) * elapsed_time
```

4. The analytical filter then updates navigation and supported calibration states together from observations and their cross-covariances. This is not a second independent bias estimate added on top of the network's full correction.
5. For selected gains, compare a fixed train-only calibrated scalar/vector against slow bounded adaptation. Gate or freeze unsupported adaptation when excitation is weak.
6. Initially train network weights offline and hold them fixed during evaluation. Outputs/hidden memory can evolve online without backpropagation or online optimizer steps. Online weight training is optional future work, not a present requirement.

This drift-based ownership is now **locked for the pilot**: the filter retains IMU biases and the network predicts their evolution. Other full-target parameter mechanisms remain open. Avoid treating a network prediction from the same innovations as an independent extra sensor. [Classical bias-state and intrinsic-calibration reference](https://docs.openvins.com/propagation.html).

Separate cadences: physical propagation follows sensor timestamps; a small network may be evaluated frequently, but bias/gain adaptation can be smoother/slower. Choose the causal history, update rate, units, bounds and cold-start behavior explicitly. Preserve the existing observability/dropout diagnostics. NCLT does not supply established bias/gain labels, so navigation loss alone does not prove uniquely correct physical calibration.

## Remaining decisions in practical order

| Decision group | Concrete choice to make | Current recommendation |
|---|---|---|
| Backend implementation details | Codebase, language, numerical pose window and update scheduling | Error-state EKF/MSCKF feature coupling already locked; no global map/loop closure initially |
| Exact error/parameter model | Bias/scale axes, priors, fixed versus estimated quantities; wheel/FOG error relations and time discretization | Persistent augmented bias states with a learned drift candidate; selected gains adapted slowly |
| Neural contracts | Inputs/history/masks, numeric bounds, cadence and reset/fallback behavior | Shared encoder/heads and exact pilot outputs/ownership locked; inputs/timing are next |
| Residual placement | Which physical transition or observation relation gets a learned mean correction | A defined wheel-model/slip residual is one supported candidate; avoid another unrestricted acceleration term indistinguishable from IMU bias |
| Training and reference | Exact recording-day split, initialization, losses, filtering through time and reference quality treatment | Held-out days and causal inputs; no test-reference-derived online features; no mandatory online weight training |
| Comparisons and success | Classical and neural comparators, sensor/head ablations, error/heading/consistency, failure behavior, latency and memory | Compare the same nominal estimator with roles off/on, using matched inputs/history; measure full frontend/backend cost on M5/16 GB |

Language/runtime, data caching and reproducibility details follow once the first estimator is specified. No additional learning-role category is required before proceeding; exact interfaces for the three agreed roles matter more than adding modules.

## What a neural model might learn

A useful schematic is

```text
parameters[k] = nominal_parameters + parameter_network(causal_history[k])
state[k+1] = physics(state[k], inputs[k], parameters[k])
             + residual_network(state[k], inputs[k], causal_history[k])
             + process_noise[k]
```

The state is the physical navigation state; inputs contain the chosen available measurements (which might be IMU/wheel measurements rather than force commands); parameters are physical/model calibration quantities, distinct from the networks' trained weights. History is causal. Learned state increments/rates must have defined units, coordinate frames and time-step scaling. Rotational increments must respect rotation geometry. This schematic is not a choice to learn every term simultaneously.

| Route | What is learned | When useful | Key design choice |
|---|---|---|---|
| **Fixed parameter identification** | Constant physical/calibration values used by a known model | A stable unknown value on one robot | Directly optimize scalar/vector parameters; a neural network is not inherently needed |
| **Neural parameter adaptation** | A small network maps history/context to bounded parameter corrections or priors | Effective calibration/slip changes, or context-dependent model behavior | Which parameters change, at what rate, and how their uncertainty enters inference |
| **Residual model learning** | An additive or otherwise structured correction to the physical transition or observation model | Missing nonlinearities or systematic effects that a few parameters cannot express | Where the correction enters, its physical frame/units, and separation from biases/noise |
| **Learn one constitutive submodel** | A force, drag, slip or other difficult relation inside otherwise analytical equations | Supporting measurements/control inputs identify that relation | Preserve known geometry/conservation equations and define the learned function's domain |
| **Learn the whole transition function** | A network predicts the state derivative/transition | A potential comparison when interpretability/generalization allow | Larger burden of validation; not the preferred first route |

These fall under **data-driven system identification**, **grey-box modeling**, **hybrid physical/learned dynamics**, and **residual dynamics learning**. Model learning and state estimation are distinct tasks, but a differentiable estimator can train a model for final state-estimation quality. One-step prediction pretraining and subsequent training through a filter are compatible. [Analytical-UKF dynamics/noise learning example](https://arxiv.org/html/2408.15610).

## Mass and inertia: feasible in the right model

Mass and inertia are meaningful parameters in equations such as m times acceleration equals force, and J times angular acceleration plus gyroscopic coupling equals torque. Learning them requires independent force/torque information or a sufficiently identified actuation model. If both thrust coefficient and mass are unknown, observations may identify their ratio rather than each separately. Known constraints, excitation and additional measurements decide what can be identified.

In ordinary IMU-driven navigation, however, the accelerometer supplies specific force. A schematic velocity equation is

```text
velocity_rate = R * inverse(accel_scale_matrix) * (measured_force - accel_bias)
                + gravity
```

It contains attitude, accelerometer scale/misalignment, bias and gravity, but **no mass**. Adding a neural mass estimate to this equation would not affect its prediction unless we changed the model to incorporate force/actuation information. Similarly, gyro-driven attitude propagation does not need a rigid-body inertia tensor. [IMU measurement and propagation models](https://docs.openvins.com/propagation.html).

NCLT documents navigation/perception channels and a differential-drive odometry model; we have not established an independent motor-command/current/torque channel for an actuator-driven identification experiment. Thus mass/inertia are valid general project ideas, with a stronger immediate data contract in a dynamics dataset such as VID or the racing-car dataset. This is about model support and identifiability, not the expressive capacity of a network. [NCLT sensors and odometry](https://robots.engin.umich.edu/nclt/nclt.pdf).

For a parameter that is constant throughout one robot's recordings, a single trainable value or a classical calibration/augmented-state estimate is an important simple comparison. A per-sample neural prediction needs a reason to vary. A useful compensation that improves state estimation can be worthwhile without claiming that it uniquely recovers physical mass or inertia.

## Better-supported initial NCLT candidates

| Quantity/effect | Possible treatment | Qualification |
|---|---|---|
| Accelerometer/gyroscope bias | Existing bias states, learned bias model, or a learned prior | Choose one coherent joint treatment rather than two independent bias compensators |
| Selected sensor scale/misalignment corrections | Constrained model parameters or slowly adapted neural priors | Planar/under-excited motion may leave some axes/parameters poorly observable |
| Wheel speed scale and left/right asymmetry | Explicit scale factors in a wheel observation or propagation model | Dataset wheel channels are already velocities; do not assume raw encoder angles or perfectly calibrated radii |
| Slip or departure from no-slip assumptions | Structured velocity/observation residual and/or time-varying wheel uncertainty | A deterministic correction and uncertainty adaptation serve different roles |
| Effective wheel geometry | A parameter of a chosen wheel-kinematic constraint | Relevant only if that parameter actually enters the selected model; it may include slip/calibration effects rather than equal true geometry |
| Unmodeled IMU-driven motion effects | A small body-frame transition/rate correction | Separate from sensor bias, timing, frame and integration errors before interpreting it physically |
| Consumer GNSS/wheel reliability | Adaptive R or robust observation weighting | Use causal inputs, validate against a fixed-noise/robust baseline and handle shared information |
| Remaining process uncertainty | Structured Q adaptation | Q does not by itself remove systematic mean error; IMU propagation already reflects acceleration from external forces |

The wheel and IMU streams must be audited as raw versus internally filtered/derived information. Some existing NCLT examples use IMU heading products and prefiltered acceleration. We can reproduce them, but should not describe the inputs as raw-only or count derived quantities as independent sensors. [Official NCLT documentation](https://robots.engin.umich.edu/nclt/), [published fusion preprocessing](https://raw.githubusercontent.com/SongJgit/KalmanNet4SensorFusion/main/data/NCLT/preprocess.py).

## Learning nonlinearities versus helping linearization

These need separate treatment:

- **Missing physics:** improve the nonlinear mean function by learning a parameterized relation or residual.
- **EKF approximation:** even a correct nonlinear mean function can be imperfectly represented by a local first-order covariance approximation.
- **Distribution/numerical errors:** Gaussian assumptions, discretization and integration contribute additional errors.

A neural residual can describe a missing nonlinear relation. Differentiate the **combined** physical-plus-learned function to obtain a consistent Jacobian. Automatic differentiation gives derivatives of the selected model; it does not eliminate curvature or make a single-Gaussian EKF exact.

A learned small matrix correction is reasonable if we explicitly define the corrected transition/measurement model and use it coherently. Directly changing F or H while leaving the mean model unchanged is a separate learned-inference approximation, with a different interpretation. It should not be confused with identifying a physical parameter. Initial recommendation: define a corrected nonlinear model, derive its derivatives consistently, and keep direct Jacobian-only corrections optional.

The error-state EKF/MSCKF backend is now locked. UKF and optimization-based alternatives remain optional comparisons, not an unresolved primary choice. Neural nonlinear expressiveness alone does not determine filter accuracy.

## Recommended progression, not additional agreed choices

1. Fix exact sensors, state vector, reference/initialization, timing and split. Dataset choice is already settled.
2. Reproduce one relevant NCLT analytical baseline and a recent learned-filter comparator under the same protocol.
3. Test a small supported parameter/calibration correction **or** one structured residual first, with a simple fixed-parameter counterpart.
4. Separately test observation/process uncertainty adaptation and evaluate consistency as well as state error.
5. Combine the components that help. Use held-out recording days and report deployment compute/history.

Start with the agreed pilot and then pursue the full supported set in stages. This is a sequencing decision, not a reduction to the pilot scope. Compensation can be redundant, and state-error improvement alone may not establish an interpretable decomposition. Parameter uncertainty can be represented by augmented states, learned priors or a justified effective noise model; exact ownership remains a design decision.

## Decisions still needed

| Decision | What remains open | Working recommendation / implication |
|---|---|---|
| First sensor contract | Detailed inertial fusion, camera default subset, dropout modes and reference protocol | IMU/FOG propagation and wheel/GNSS/visual inputs agreed; magnetometer disabled and barometer excluded. Configurable one-to-six-camera/all-pairs design agreed |
| State and geometry | Exact augmented-state/calibration representation and visual nuisance quantities | Five planar navigation outputs and tilt-aware internal attitude are **locked**; selected error categories agreed, exact augmentation open |
| Nominal model and backend details | Exact state/observation/error equations, joint noise and implementation | Error-state EKF/MSCKF and IMU/FOG propagation plus wheel/GNSS/visual updates locked; detailed fusion/noise equations remain open |
| Learned quantity | Exact sensor-adaptation, model-residual and uncertainty outputs | All three logical roles **agreed**; select small data-supported outputs and ownership before network architecture |
| Time behavior | Fixed, slow adaptation or per-step correction; permitted causal history | Prefer stable parameters/slow adaptation for calibration, faster residuals only where justified |
| Training | Direct prediction/calibration training, training through the filter, sequence length and loss | State-estimation performance remains the actual target; visual weights remain frozen initially |
| Evaluation/reference | Exact sessions, reference quality/coverage, initialization, alignment, intervals and metrics | Published 22/2/3 recording-day split is a concrete proposal, not yet a locked experiment protocol |
| Comparators | Exact classical and neural systems; matched sensor sets and timing | Include an analytical fixed-parameter baseline plus relevant published learned filters |
| Numerical targets | Useful error levels, failure/recovery behavior, covariance consistency, compute/history/latency | Measure on M5/16 GB before setting runtime/training-budget promises |

No novelty search or publishability gate is required to proceed. No estimator has been implemented or evaluated. One small public sensor archive has now been inspected for the sensor-format audit; no image/LiDAR payload or full dataset has been staged.
