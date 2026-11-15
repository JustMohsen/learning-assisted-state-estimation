# State-estimation project: goals and decision record

> **Current project definition: [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) and [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)** (14 November 2026; vision now out of scope). Earlier consolidated design, archived: [PROJECT_SNAPSHOT.md](../archive/PROJECT_SNAPSHOT.md). This log retains chronological discussion; the snapshot and later accepted decisions supersede older proposals/open-status wording.

Created: 13 November 2026. Updated: 14 November 2026. **Backend locked: error-state EKF with tightly coupled MSCKF-style visual tracks, bounded recent rig poses and no persistent map/loop closure initially.** NCLT, five planar outputs, tilt-aware attitude, IMU/FOG propagation plus wheel/GNSS/visual updates and three learning roles remain agreed. Frozen SuperPoint/LightGlue and semantic masks, camera configurability and exclusions persist. Pilot ownership, GRU/input timing/memory, startup/recovery, baseline mathematics, initial numerical/visual settings and training goals are locked. **Design decisions are consolidated in the project snapshot; optional post-pilot refinement and deferred Gemma experiments are recorded.** Exact initial camera pair needs geometry/image verification; distance alone is not an optimality guarantee. Lateral no-slip investigation is subsequent work within this project; deeper linearization/data-driven modelling is the final design discussion. No estimator implementation has begun. Use plain-text equations.

## North star

Develop a physical state estimator that improves on a relevant state-of-the-art system, or achieves comparable accuracy with a meaningful advantage such as lower compute, more causal/less delayed inference, less history, or fewer sensors. Success is an empirical research target, not a promised outcome. **Start with NCLT ground-robot navigation.** VID indoor and the Węgrzynowski et al. IROS 2024 racing-car dataset are optional future investigations, not required additional datasets.

The user explicitly does **not** require novelty, a publishability argument, or a search for something unclaimed. Continue with the approach we find promising and interesting. Use prior work for useful ideas and honest comparisons, without turning novelty into a decision gate.

Learned physical residuals, sensor/actuator biases and uncertainty adaptation are candidate tools for improving the estimator. Estimating a force, wind value, sideslip or other auxiliary quantity is not an independent project objective unless it contributes to the state-estimation target. A dedicated computer-vision model is also optional.

Visual-training scope is now fixed: use existing classical methods and/or **frozen pretrained** visual models. The initial camera route is **SuperPoint + LightGlue with frozen weights**. Training a compact visual model from scratch and developing a new general VO/SLAM system are excluded. Visual fine-tuning is deferred and optional. Small learned estimator/reliability/fusion heads remain possible; they are not automatically required after a frozen model. Kalman gain and state correction remain analytical; direct neural gain prediction is outside the primary architecture.

The initial dataset/application family is settled. Decisions should reflect discussion with the user; a provisional recommendation about sensors, state or backend is not an agreed selection.

## Decision goals

| Goal | Decision to make | Completion condition | Current status |
|---|---|---|---|
| 1. Application and physical estimation problem | NCLT Segway planar navigation | Agreed physical use case and why better estimation is valuable | Platform and five navigation outputs agreed; detailed operating/failure targets follow |
| 2. Dataset and specific experiments | NCLT first; exact sessions and evaluation protocol | Dataset agreed, usable channels/reference/baseline audited and exact experiments specified | **NCLT locked** for development/training/evaluation. Session split and baseline/data execution remain open; VID indoor/racing-car data optional future work |
| 3. State and useful performance | Exact state vector; useful accuracy, latency, history, compute and sensor requirements; auxiliary states only as justified | State vector and measurable accuracy/efficiency targets agreed, with reference support | **Five reported quantities locked:** north/east position, north/east velocity and heading. Internal 3D attitude and sensor-error categories agreed; exact auxiliary states and numerical targets open |
| 4. Sensor and reference contract | What enters the estimator, what is training-only, what is evaluation-only; origin and uncertainty of reference | Explicit channel/provenance table; reference limitations and information shared with estimator documented | Wheel/GNSS/frozen-visual inputs agreed; magnetometer disabled, barometer excluded. One-to-six-camera/all-pairs design and removal diagnostics agreed. Exact FOG fusion, default view set and reference protocol open |
| 5. Nominal physics and learning | Physical parameters, calibration, residual/nonlinear functions, Q/R and approximation error supported by NCLT | Justified model/learning design plus ablations; physical parameters and uncertainty are treated coherently | Error-state EKF/MSCKF backend, IMU/FOG propagation and three learning roles locked. **Calibration ownership/network interfaces active now**; exact parameter lifecycle, residual placement and outputs open |
| 6. Comparisons and experimental questions | Classical baseline, relevant published systems, learning ablations and useful tradeoffs | Comparable inputs, initialization, latency and training information; informative experiments | Exact methods open. **Novelty is not a completion condition** |
| 7. Evaluation and feasibility | Splits, generalization, calibration, realistic degradations, runtime/compute, data volume and reproducibility | Written protocol and feasible first experiment; implementation starts only after the problem is sufficiently defined | Hardware confirmed by user: MacBook Air M5, 16 GB RAM. Local/cloud feasibility being assessed |

The seven goals expand the user's application, dataset, and approach/comparison decisions. They are decision gates, not a commitment to test every estimator family.

## Preferences and evidence that must persist

- Scientific objective: develop a better state estimator than a relevant state-of-the-art comparator, or comparable estimation with lower compute, less history/latency, more causal operation, or fewer sensors. Auxiliary physical quantities are possible means to that objective.
- First scope: NCLT Segway ground-robot navigation. VID indoor and the 2024 racing-car data are optional future investigations. Other application searches are parked.
- Novelty/publication is not required. Do not keep reopening a novelty or dataset-choice gate; evaluate the best promising approach we can build.
- Preferred direction: nominal physics plus a learned residual, potentially combined with sensor/actuator bias and covariance learning. Modeling disturbance covariance without estimating a physical wind value is of interest.
- Hardware: the user reports a MacBook Air M5 with 16 GB RAM. No local training throughput or numerical benchmark has been measured.
- Visual training from scratch and development of a new general learned VO/SLAM system are excluded. Visual fine-tuning is deferred and optional. Classical features, frozen pretrained visual models and small estimator heads remain candidates.
- Initial visual route: frozen SuperPoint + LightGlue. Optional learned alternatives: XFeat + LighterGlue and ALIKED + LightGlue. Optional classical alternatives: ORB + Hamming matching and Shi–Tomasi + pyramidal KLT. These investigations and fine-tuning may never be undertaken; they are not completion requirements.
- The user is interested in classical/learned visual complementarity, physical sensor/actuator gain/scale modeling, mass/inertia where identifiable, and learned nonlinear/model residuals. Keep Kalman gain/correction analytical. Backend/visual coupling is now locked; specify exact sensor/model-residual/uncertainty ownership and interfaces. Nonlinear/data-driven modeling remains part of this discussion.
- **Navigation outputs locked:** north/east position, north/east velocity and heading. **Internal tilt-aware 3D attitude agreed.** Exact auxiliary/calibration states and any vertical nuisance treatment remain open; do not reopen the planar output decision.
- Agreed inputs: wheel speeds and receiver-processed consumer GNSS; no satellite-level GNSS reconstruction. Camera processing agreed through frozen SuperPoint/LightGlue. Skip barometer.
- Agreed: disable magnetometer initially, potentially indefinitely; adding it later is not required.
- Agreed design capability: select any one through six cameras; support all distinct camera pairs and a reduced useful-pair mode for compute/lag comparisons. Six cameras have 15 distinct simultaneous pairs; this is a design capability, not a benchmarked implementation.
- Agreed sensor-removal diagnostics: errors for unsupported configurations/no valid propagation fallback, and degraded-mode/uncertainty warnings for supported but weakened estimation. Excitation/scene dependence prevents a simple sensor-list guarantee of observability.
- Agreed propagation family: combine IMU and FOG within the inertial subsystem, retaining separate physical error models and consistent covariance, then apply wheel/GNSS/visual updates. Detailed interval/fusion mathematics remains open.
- Agreed sensor-error categories: selected IMU biases/scales, FOG bias/scale and wheel calibration. The user's question about initialization followed by neural estimation at each step is an open mechanism/cadence question, not a decision to overwrite every parameter independently every sample.
- Agreed learning roles: sensor adaptation, model residual and uncertainty adaptation. This does not choose the number of networks, exact outputs or a requirement to learn every covariance/axis.
- Mapping decision: bounded local visual geometry/rig-pose history with MSCKF feature constraints; **no persistent map/loop closure initially**, locked by the user. Closest reviewed numerical learned-filter NCLT experiments do not perform SLAM; broader NCLT research does.
- Learned architecture agreed: one small shared GRU for numerical history, separate sensor-adaptation/model-residual/process-uncertainty heads and small GNSS/wheel/visual quality heads. The [ownership note](./calibration-ownership.md) records locked pilot output dimensions/ownership and execution clocks; hidden/head sizes and numerical limits remain under discussion. No verified actuator command/current/torque channel has been established for NCLT; wheel residuals must not be labeled actuator gain. Environmental Q describes residual-model uncertainty, not uniquely identified wind covariance.
- **Pilot/full-target scope locked:** first learned experiment is IMU bias evolution, one forward wheel-model residual and observation uncertainty; the full supported set is explicitly planned for pursuit, including gains, FOG adaptation and process/sensor-noise adaptation. Pilot staging must not quietly become the final scope. Unsupported/unidentifiable components require evidence and discussion rather than silently abandoned or unconstrained outputs.
- Wheel-slip policy locked: continuous reliability/covariance inflation, severe-outlier rejection, bounded mean-error compensation when useful, and a diagnostic suspected-slip event with recovery hysteresis. Exact numerical thresholds and implementation details remain open. NCLT slip-event ground-truth labels have not been established.
- Latest visual decision: the user's moving-bus/dominant-foreground case warrants **semantic instance masking in the baseline**, alongside RANSAC, robust geometry, IMU/FOG-prior motion checks and bias-consistency protection. The user explicitly favors DynaSLAM-style masking and adoption of DynaVINS/DynaVINS++ principles. Use frozen pretrained segmentation; Mask R-CNN is the first reference, exact checkpoint/runtime open. This supersedes the earlier optional-mask recommendation. See [architecture options](./architecture-options.md) for evidence and implementation boundaries.
- The user wants sensor-frame IMU/FOG bias, gain and noise modeling preserved. Calibration inside the joint estimator and missing-input fallback/observability treatment remain explicit design requirements.
- Presentation preference: LaTeX equations appeared unrendered. Use plain-text equations/code blocks in replies and maintained notes; existing display equations have been converted.
- The intended task is physical state estimation with meaningful sensor correction; a drifting displacement predictor alone is insufficient.
- The user previously tried EuRoC and observed meter-to-tens-of-meters errors, including easy sequences. This is the user's experiment report, not a reproduced result.
- The user is dissatisfied with that error magnitude for a small drone. We have not yet agreed on a numerical replacement target.
- The user clarified that the prior EuRoC experiment excluded cameras and was effectively dead reckoning over the full trajectory. Further debugging or reconstructing the old experiment is out of scope.
- The user is not opposed to EuRoC; their concern is vigilance about drift and practical accuracy. EuRoC remains eligible where its sensors and physical-model data match the chosen task.
- The initial literature shortlist emphasized general navigation and sensor fusion. Residual rigid-body dynamics requires a fresh check for control/actuator data and physical parameters.

## Modeling context and historical dataset search

The initial dataset decision below has been resolved in favor of **NCLT**. Earlier candidates and access audits are preserved as background/future-work evidence, not active competing primary datasets. See [NCLT sensors and planar-state proposal](./nclt-sensors-and-state.md) for the current discussion and [NCLT modeling options](./nclt-modeling-options.md) for the next discussion.

### The central physical distinction

For **actuator-driven dynamics**, a nominal model uses quantities such as motor speeds, thrust, steering, mass and inertia. A residual can describe aerodynamic/tire forces, actuator mismatch or other unmodeled dynamics. The dataset must support the inputs and intended output claims.

For **IMU-driven kinematic propagation**, measured specific force already reflects the effect of external forces on the vehicle. Residuals can represent bias, scale, timing or integration/model errors. Adding a separate wind acceleration to that propagation without a consistent measurement model can double-count effects. This does not invalidate residual learning, but changes its physical interpretation.

The generic additive form is only schematic:

```text
next_state = physical_model(current_state, inputs)
             + learned_residual(causal_inputs) + process_noise
```

Orientation corrections should respect rotation geometry. Force, sensor-bias and model-residual states may be confounded unless the measurement setup and excitation distinguish them. Learning R does not automatically identify or remove a physical bias.

### Candidate pairs and present evidence

| Application / physical problem | Candidate data | Why inspect it | Unresolved issue before selection |
|---|---|---|---|
| Indoor drone: navigation plus model mismatch or disturbance estimation | **VID indoor sequences** | Visual/inertial data, actuation/dynamics channels, mocap pose; selected rope-pull trials have force sensing | Authenticated inventory and one recording's download headers verified. Eight indoor bags total about 68 GB; filename durations total about 15.2 minutes. Payload channels, force calibration and evaluation still need audit |
| Indoor drone: aerodynamic residual or force estimation | **NeuroBEM** | Motor data, physical parameters and motion-capture-derived reference; established hybrid-model comparisons | Agile motion; processed derivatives mix onboard and mocap information; does not by itself provide the same externally aided navigation task as a UWB dataset |
| Indoor drone: numerical sensor fusion and reliability/bias adaptation | **UTIL / MILUV** | External UWB observations, mocap pose and supplied classical fusion baselines | Actuator-driven rigid-body residual experiment is not established by their listed channels; controls/parameters require audit |
| Road vehicle: dynamics-state estimation with a learned model residual | **KIT Multi-Surface Driving Maneuvers**; **UrbanNav / Boreas** remain navigation alternatives | KIT provides steering/dynamics signals, vehicle parameters and optical velocity instrumentation; navigation datasets answer a different physical question | KIT raw channel provenance, usable operating regimes and classical/learned baseline availability require audit before choosing |
| Outdoor drone: dynamics-aided navigation | **VID outdoor / NTU VIRAL** | VID has dynamics channels and RTK; NTU has UWB, vision, LiDAR and external tracker positions | Different reference quantities and independent truth coverage; external forces are not labeled throughout outdoor data |
| Off-road ground vehicle: terrain-dependent dynamics | **TartanDrive / TartanDrive 2.0** | Action-conditioned, multimodal off-road data is physically relevant to slip and terrain effects | Dataset family/version, exact controls, pose-reference provenance and an estimation rather than prediction protocol need audit |
| Spacecraft: relative navigation and dynamical model mismatch | **SHIRT** | Temporal image observations and simulated/testbed trajectories; published adaptive neural/UKF work | Mainly vision and limited trajectory diversity; reference is HIL/simulation rather than an independently instrumented space mission |
| Planetary rover, marine surface craft, underwater vehicle | Further targeted search needed if preferred | Strong physical models, disturbances and navigation motivations | The existing review has not established a public dataset meeting control-input, reference, access and baseline requirements together |

This table preserves the broader search history. **NCLT is now selected first.** VID indoor and the IROS 2024 racing-car dataset are explicitly optional future investigations. These historical rows do not reopen the primary dataset choice.

### Newly relevant candidate: VID

The 2021 VID paper describes visual, inertial and dynamical measurements, control inputs, motor speeds/currents, and six-axis force data. The official sequence table distinguishes indoor mocap trials, outdoor/night trials with RTK, and selected force-sensor experiments. Hover, circular, figure-eight, loading and rope-pull cases make it relevant to more than aggressive trajectory tracking. Reference coverage must be checked per sequence. [VID paper](https://arxiv.org/abs/2103.11152), [official dataset and sequence table](https://github.com/ZJU-FAST-Lab/VID-Dataset).

**HDVIO** (RSS 2023) and **HDVIO2.0** (T-RO 2025) are relevant references for hybrid physical/learned dynamics integrated with visual-inertial estimation and disturbance estimation. Their results support investigating this direction. The papers use multiple datasets; their outdoor VID treatment includes an offline visual-inertial SLAM training reference. A dataset's RTK channel and a particular paper's chosen training reference are not the same thing. [HDVIO](https://arxiv.org/abs/2306.11429), [HDVIO2.0 paper](https://rpg.ifi.uzh.ch/docs/tro25_Cioffi.pdf), [implementation](https://github.com/uzh-rpg/hdvio2.0).

### Feasibility audit: ground-vehicle alternatives and VID access

Audit date: 13 November 2026 (Toronto). This pass inspected publisher metadata, a small README, portal file listings and download headers; it did not reproduce an estimator or download sensor archive bodies. A normalized [access-evidence record](./dataset-access-audit.json) preserves the checks without session cookies or passwords.

| Candidate | New verified evidence | What remains unresolved |
|---|---|---|
| **VID** | Published shared password accepted; authenticated listing exposes indoor, motortest, night and outdoor folders. Eight indoor recordings are listed. A loadless-hover recording returns HTTP 200 with the expected filename and 5,879,866,888-byte content length | Sensor contents, timestamps, calibration and force-reference channels remain uninspected; successful headers are not a successful full recording decode |
| **KIT Multi-Surface Driving Maneuvers** | Public metadata documents optical velocity, CAN/steering signals, MAT data and physical parameters. Normal public download route returns HTTP 200 and a TAR attachment | Need actual per-channel source definitions, quality flags, usable acceleration/heading inputs and baseline audit; archive body not downloaded |
| **TU/e Driving Data of a Real F1tenth Car** | The small Zenodo README was retrieved successfully; it documents commanded velocity and Vicon-derived velocity streams in nine ROS bags | Its documented command/response focus does not establish the complementary onboard sensor suite needed for the proposed state-estimation task |
| **Węgrzynowski et al. F1TENTH estimation data** | The 2024 paper documents IMU, RPM/current, steering, motion-capture pose and derived velocity, with classical/hybrid/neural estimator comparisons | The specific archive/code endpoint was not established in this pass. Do not confuse it with the unrelated TU/e dataset |

**KIT Multi-Surface Driving Maneuvers** was published on 30 April 2026. Its metadata lists 41 runs, 42 minutes, 1000 Hz synchronized recording and a 171.7 MB archive, with a parameter script and reference-frame offsets. A possible task is estimating lateral velocity/sideslip while withholding optical-velocity channels for evaluation. Tie-rod forces are not direct tire-contact force labels, and different surfaces do not by themselves provide friction-coefficient truth. [Institutional record](https://publikationen.bibliothek.kit.edu/1000192628), [archive and signal metadata](https://www.radar-service.eu/radar/en/dataset/44a91t97pmnha1k9).

The more immediately relevant ground-vehicle prior art is *Learning dynamics models for velocity estimation in autonomous racing* (IROS 2024). It compares a physical single-track/tire model, a learned residual, neural tire models, and recurrent estimators; it also studies learning through a UKF and adapting to tire changes. Its motion-capture-derived velocity reference is explicitly differentiated/smoothed rather than independently measured velocity. It is a dynamics-state estimation task; it does not claim globally drift-free position from proprioception. [Paper and dataset description](https://arxiv.org/html/2408.15610).

The TU/e dataset is useful as an example of why public data plus mocap is not sufficient by itself: its README primarily documents commanded versus actual velocity for dynamics identification. Treating the reference velocity as an available sensor would change the estimation problem. [Dataset](https://zenodo.org/records/12536536), [small README](https://zenodo.org/records/12536536/files/ReadMe.md?download=1).

**Consequence for goals 1–2:** compare the physical task as well as the platform. Indoor-drone navigation/disturbance estimation and road-vehicle velocity/sideslip estimation have different outputs and deployment value. Either can fit residual physics, but choosing the lower-dimensional task is a user decision, not a substitute for an agreed navigation goal.

**VID scale and access qualification:** the eight listed indoor bags sum to 67,998,366,141 bytes (about 68.0 GB), with individual bags about 5.88–11.57 GB. Their filename duration labels sum to 913.68 seconds (15.228 minutes). Those are inventory/filename calculations, not decoded recording measurements. The short total and small number of trials constrain what training/test generalization claims can be made. High sample rate and large image files do not create many independent operating conditions.

**Decision dependencies now resolved:** the scientific outcome is state-estimation performance, with physical quantities serving as potential auxiliary tools, and **NCLT is the first dataset**. Do not ask the user to choose again between a navigation project and standalone disturbance estimation, or between primary datasets.

### Modeling, vision and computing discussion

The [modeling and compute options note](./modeling-and-compute-options.md) records mean residuals/covariance roles, UWB, computing feasibility and earlier cost context. The current [architecture discussion and flowcharts](./architecture-options.md) addresses frozen-feature outputs, visual observability, estimator structure, shared-image correlation, candidate pretrained models, DROID-SLAM, actuator versus Kalman gain, and model/linearization error.

Current handoff: [comprehensive project snapshot](../archive/PROJECT_SNAPSHOT.md). The full-set rollout order is accepted. Evaluation/comparisons and Python/PyTorch/local-first implementation recommendations are accepted. The [training protocol](./training-protocol.md) is accepted as an adjustable starting point, alongside the earlier backend/pilot policies. Begin with two cameras; one-to-six/all-pairs is the broader design. Exact camera IDs need an overlap/triangulation audit. Loss weights, calibration terms, physical bounds and thresholds remain tunable. Lateral no-slip investigation belongs to subsequent project stages; deeper linearization/data-driven modelling is reserved as the final design decision. No estimator implementation has started.

### Current decisions and open questions

| ID | Item | Status |
|---|---|---|
| D1 | Improve state-estimation accuracy or achieve comparable accuracy with useful deployment advantages | Agreed |
| D2 | First application/platform | NCLT Segway ground-robot navigation; other domains remain background/future work |
| D3 | Classical visual processing and frozen pretrained visual processing are allowed | Agreed |
| D4 | Train a visual model from scratch or create a new general learned VO/SLAM system | Excluded |
| D5 | Fine-tune visual weights | Deferred, optional |
| D6 | Initial camera feature/matching route | Frozen SuperPoint + LightGlue selected; exact implementation and CPU/MPS feasibility remain open |
| D7 | Kalman gain and state-correction calculation | Keep analytical/mathematical; direct neural K prediction is reference material, outside the primary architecture |
| D8 | First development, training and evaluation dataset | **NCLT — locked by the user** |
| D9 | Novelty, publication or finding an unclaimed architecture | **Not required; not a decision gate** |
| D10 | VID indoor and the IROS 2024 racing-car dataset | Optional investigations/future work; may never be undertaken |
| D11 | Consumer GNSS input | Use released receiver-processed fixes; user explicitly does not want raw GNSS reconstruction |
| D12 | Wheel input | Use released left/right wheel speeds |
| D13 | Barometer | Skip |
| D14 | Magnetometer | Disabled initially and potentially indefinitely; no required later investigation |
| D15 | Camera count and pair-selection capability | Support one through six cameras, all-pairs mode and useful-pair mode; initial subset/implementation/runtime untested |
| D16 | Sensor-removal diagnostics | Modular availability with errors for unsupported configurations/no propagation fallback and appropriate degraded-mode/uncertainty warnings |
| D17 | High-level nominal estimator family | IMU/FOG-assisted inertial propagation with wheel/GNSS/visual updates agreed; detailed fusion/noise treatment open |
| D18 | Equation presentation | Plain-text/code-block equations; LaTeX did not render in the user's interface |
| D19 | Navigation outputs | North/east position, north/east velocity and heading: five quantities, full 2D navigation |
| D20 | Internal attitude | Retain tilt-aware 3D attitude for inertial/visual geometry |
| D21 | Sensor-error scope | Selected IMU biases/scales, FOG bias/scale and wheel calibration agreed; exact parameterization/ownership/cadence still open |
| D22 | Learning roles | Sensor adaptation, model residual and uncertainty adaptation agreed; shared-encoder/head organization now selected in D27 |
| D23 | Semantic/instance masking for dynamic scenes | Include frozen DynaSLAM-style masks in the intended baseline; Mask R-CNN is the first reference, exact checkpoint/runtime open. Retain an ablation switch; no new visual-model training required |
| D24 | Inertial-prior visual robustness | Adopt IMU/FOG-assisted motion consistency and DynaVINS/DynaVINS++ calibration-consistency principles; exact gating, robust update and recovery adapted to the chosen backend |
| D25 | Backend and visual coupling | **Locked:** error-state EKF with tightly coupled MSCKF-style feature-track updates and bounded rig-pose history |
| D26 | Initial mapping/history policy | **Locked:** temporary 3D feature geometry; no persistent map or loop closure initially |
| D27 | Learned architecture organization | **Locked:** one small shared GRU with separate calibration/model-residual/process-uncertainty heads; small GNSS/wheel/vision observation-quality heads. Exact size/numeric interfaces open |
| D28 | Pilot and full research target | **Locked:** limited initial learned experiment is a pilot; plan to pursue/evaluate the full supported bias/gain/model-residual/process/measurement uncertainty set in stages |
| D29 | Wheel-slip handling | **Locked:** continuous reliability/covariance inflation, bounded mean correction when useful, hard rejection for unusable observations, suspected-slip diagnostic and hysteretic recovery. Numerical gates remain to tune |
| D30 | Exact pilot outputs and ownership | **Locked as proposed:** six IMU bias-drift outputs with filter-owned bias states; one forward wheel residual in m/s; two GNSS variance multipliers; one wheel variance multiplier; one multiplier per accepted visual feature-observation block. Pilot gains/wheel scale and general process noise fixed; FOG bias analytical slow state |
| D31 | Encoder input groups | **Locked:** numerical sensor history plus previous estimator context/trusted innovation summaries, with input availability/age |
| D32 | Observation-head invocation | **Locked:** invoke GNSS/wheel/visual quality heads when their observations arrive, not at every propagation tick |
| D33 | Recurrent memory | **Locked:** persistent causal forward GRU; exact hidden size/training truncation remains open |
| D34 | Neural execution timing | **Locked:** shared GRU and bias head on IMU events with elapsed time; wheel residual/quality, GNSS quality and visual quality on their observation events using common timestamp alignment |
| D35 | Training batches and inference backend | **Locked:** mini-batches of independent sequences for training; online error-state EKF/MSCKF inference retained. No joint nonlinear window optimizer selected |
| D36 | Initialization | **Locked:** controlled common initial-pose prior for the pilot, disclosed and shared with baselines; separate sensor-only startup mode. Velocity/calibration from supported evidence/priors, nonzero covariance, zero GRU memory and causal baseline warm-up; no ongoing evaluation-reference feedback |
| D37 | Resets and recovery | **Locked:** retain navigation/calibration/GRU through ordinary sensor gaps; gate returning observations; reset affected visual tracks locally. Coordinated full reinitialization for independent sequences or unrecoverable propagation/timing/numerical failure; log losses/resets and do not recover from evaluation ground truth. Contiguous training chunks retain memory with truncated gradients |
| D38 | Baseline estimator mathematics | **Locked:** 3D internal position/velocity/quaternion plus IMU biases and FOG rate bias, planar reported outputs, NED/IMU-centered propagation with mounting transforms, midpoint integration, interval IMU/FOG fusion, agreed observations, Joseph-form covariance update/reset, and conditional-GRU uncertainty approximation. Detailed derivations and numerical verification remain implementation work |
| D39 | Lateral wheel no-slip constraint | **Committed within this project:** implement/investigate a soft lateral constraint after the forward-speed pilot, compare on/off, and relax/reject it during suspected lateral sliding. Not relegated to optional future work |
| D40 | Linearization and data-driven modelling discussion | Baseline policy accepted; reserve the deeper discussion of linearization, learned model residuals/submodels/parameters and identifiability as the **last design decision**, after the other decision groups. No enhanced method selected yet |
| D41 | Starting recurrent configuration | **Accepted user preference:** at least two GRU layers; start with two layers and 64 hidden units per layer. Network configuration remains adjustable during development |
| D42 | Learned-output initialization and arithmetic | **Accepted:** learned mean corrections start at zero, observation-variance multipliers at one; Float64 filter/covariance and Float32 learned module |
| D43 | Observation-variance bounds | Bounded-positive policy accepted with explicit severe-slip rejection; initial 0.25–25 variance range remains a tunable starting point per sensor. Signed mean parameterization accepted in D44 |
| D44 | Initial numerical/network settings | **Accepted working defaults:** two GRU layers of 64 units, small separate heads with one 32-unit hidden layer, scaled-tanh signed means (centered sigmoid equivalent), bounded positive observation-variance multipliers, baseline initialization and agreed precision. Physical limits/noise/gates and configuration tune on training/validation |
| D45 | Neural training goal | **Accepted:** backpropagate through the analytical ESKF/MSCKF over sequences to optimize navigation error, with justified regularization and uncertainty-calibration assessment/term. Include position, wrapped heading and reliable derived velocity. Exact loss form/weights, calibration objective and full data protocol remain open |
| D46 | Regularization during degradation | **Accepted caveat:** head-specific/event-aware penalties; permit fast wheel-residual and observation-uncertainty responses or rejection during supported slip/blur/obstruction. Physical bias adaptation remains protected; skip missing visual updates, use guarded recovery, and retain navigation loss on valid reference intervals |
| D47 | Initial visual operating configuration | **Accepted adjustable starting point:** two cameras, temporal plus useful simultaneous matching, 800-pixel longest edge, up to 1024 retained background features/image, frozen Mask R-CNN ResNet-50-FPN COCO_V1 at each processed approximately 5-Hz image, ten rig-pose clones, up to 200 accepted tracks/rig update and cached frozen frontend outputs. One-to-six/all-pairs remains design capability rather than the starting workload. Actual pair selection/benefit of a third camera require geometry/image verification; a larger baseline alone does not prove optimal depth |
| D48 | Training protocol | **Accepted adjustable starting point:** fixed 22/2/3 recording-day split and dates, ten-second truncated gradient chunks with two independent stateful streams, weighted normalized navigation MSE, agreed regularization/calibration checks, staged mean/noise/joint training, initial AdamW/clipping settings, full-validation stopping and three-seed final comparisons. Refine implementation/numerical details while building |
| D49 | Evaluation and comparisons | **Accepted:** declared navigation/drift/robustness/consistency/compute metrics; identical learning-disabled comparator; first external candidates OpenVINS and NCLT KalmanNet under matched or explicitly qualified protocols; per-head/combined ablations, fair initialization/history/alignment and complete failure accounting |
| D50 | Implementation and compute | **Accepted:** Python/PyTorch prototype with one analytical training/replay core; reference/model reuse and incremental numerical-to-visual-to-learning integration; CPU Float64 filter, profiled Float32 CPU/MPS modules, incremental data/caches, local-first profiling and measured need for later acceleration/cloud |
| D51 | Full-set rollout order | **Accepted:** analytical baseline → locked pilot → structured process uncertainty → soft lateral wheel constraint → FOG bias/interval-noise adaptation → common wheel and supported IMU/FOG gains → richer covariance blocks/full-set combination. Investigate separately/jointly, check identifiability and retain negative findings; full supported scope remains committed |
| D52 | Deferred Gemma 4 experiments | **Accepted future-experiment order:** frozen offline annotation/teacher on training clips first, then low-frequency supplemental scene context with causal timing/age/fallback. Existing geometric frontend and initial estimator stay the baseline; no mandatory Gemma inference or new general vision-model training |
| D53 | Post-filter neural refiner | **Accepted as a potential experiment/future work after the pilot**, not a required initial component. Small bounded causal readout; initially freeze the core, compare raw/refined outputs without recursive feedback and separately validate uncertainty. Detailed design remains adjustable |
| O1 | Backend implementation and sensor derivations | Python/PyTorch is the accepted first prototype, OpenVINS a reference/external candidate. FOG colored-noise treatment/Jacobians, correlated-noise derivations, adapters and numerical verification remain implementation work |
| O2 | Other visual methods or conditional fallback | Optional: XFeat + LighterGlue; ALIKED + LightGlue; ORB + Hamming; Shi–Tomasi + pyramidal KLT. No commitment to run all or any of them |
| O3 | Detailed neural inputs and full-target expansion | Input groups, GRU, execution clocks and persistent memory locked. Exact vector construction/size/bounds and later-stage outputs remain open |
| O4 | Model mismatch, linearization and additional neural correction | Nonlinear nominal propagation/measurement versus linearized error/covariance clarified. Existing grey-box baseline retained. A small causal post-filter residual readout is proposed; it predicts remaining estimator error rather than a uniquely identified nonlinear component |
| O5 | Experimental questions and useful numerical targets | Open; does not require a novelty claim |
| O6 | Sensor availability and matched-subset comparisons | Capability/diagnostics agreed; concrete fallback modes, training masks and runtime observability/initialization checks still to define |
| O7 | Visual-history and camera implementation details | Starting ten-clone/1024-feature/200-track configuration accepted and tunable. Verify exact camera IDs, overlap/triangulation, delay and track-consumption policy; third-camera value unmeasured. No initial global map/loop closure |
| O8 | Detailed moving-object implementation | Frozen Mask R-CNN ResNet-50-FPN COCO_V1 with per-processed-image masking accepted. Runtime/operator compatibility, class/confidence/boundary defaults, quality gates and covariance-consistent calibration/recovery details require implementation verification |
| O9 | Wheel-slip numerical implementation | Policy locked in D29 and lateral-constraint investigation committed in D39; gate thresholds, recovery timing, lateral-constraint details, optional per-wheel extensions and event-validation details remain open |
| O10 | Gemma 4 execution details | Future-experiment ordering accepted in D52. Dataset/sample, prompt/label validation, frozen E2B/E4B runtime, sampling cadence and measured benefit/cost remain deferred implementation details |
| O11 | Post-filter refiner details | Optional post-pilot experiment accepted in D53; architecture size, causal inputs, training residual protocol and refined-output uncertainty remain deferred experimental details |

Working recommendation, not an agreed complete architecture: use a transparent estimator with the selected frozen visual route, and add small learned physical/uncertainty components inside the recursion. If optional front-end comparisons are pursued, compare them individually before testing dual processing. A fixed-parameter analytical estimator using SuperPoint/LightGlue is an important ablation, but must not be labeled a fully classical vision pipeline. A relevant classical state-estimation comparator is still needed for the scientific claim; the optional front-end sweep is a separate question.

The user accepted evaluation/comparison policy, Python/PyTorch/local-first implementation and full-set rollout order along with the earlier decisions. **Current discussion: model mismatch, linearization/data-driven modelling and Gemma 4 feasibility.** The [discussion note](./model-learning-and-gemma4.md) provides a brief conceptual recommendation and source-checked VLM capabilities/costs; Gemma's deferred offline-then-low-frequency experiment order is accepted; the post-filter refiner is accepted as a potential post-pilot experiment/future work.

## Remaining decisions in order

This checklist supersedes earlier discussion-order suggestions. Locked dataset, output state, backend/coupling, sensor exclusions, visual route, dynamic-scene policy, network organization, slip policy and pilot outputs are not reopened. Concrete numerical hyperparameters can be selected and tuned during implementation/validation; they need not all become separate user-approval gates.

| Order | Remaining decision | Specific unresolved items |
|---|---|---|
| 1 | Network inputs and timing/history — architectural choices locked | Sensor-plus-estimator input groups, IMU-triggered GRU, observation-event heads and persistent causal memory agreed; exact vector and numeric details are implementation work |
| 2 | Initialization, resets and recovery — policy locked | Controlled common pilot pose prior versus separately labeled sensor-only startup; conservative calibration/covariance priors, baseline GRU warm-up, ordinary gaps/local visual resets/full failure resets agreed. Numerical thresholds remain implementation work |
| 3 | Baseline mathematics — model family locked | Internal 3D state/vertical nuisance quantities, interval IMU/FOG fusion, frame/lever-arm observation models, covariance update/reset and conditional-GRU approximation agreed. Exact derivations verified during implementation. Lateral-constraint investigation committed subsequently; deeper linearization/data-driven modelling reserved for item 10 |
| 4 | Numerical/network starting configuration — accepted | Two-layer/64-unit GRU, small heads, bounded signed means and positive variances, initialization and precision agreed. Event-aware regularization accepted. Physical output bounds, baseline-noise calibration, exact gates/recovery and tuning remain implementation work |
| 5 | Visual starting configuration — accepted | Two-camera starting workload and numeric/checkpoint/cadence/cache defaults accepted in D47. Exact IDs/overlap/triangulation, possible third-camera benefit, runtime, delay, track handling and numerical tuning are implementation work. Broader one-to-six/all-pairs capability retained |
| 6 | Training and data protocol — starting recommendations accepted | Exact day split and initial stateful chunk/batch/optimizer/loss/staging/stopping protocol accepted in D48. Detailed calibration objective, reference-quality processing, tuning and differentiable implementation remain engineering work |
| 7 | Evaluation and comparisons — policy accepted | Metrics/comparator/ablation/fairness/robustness/compute policies accepted in D49. Exact adapters, numerical targets and reference-quality implementation remain engineering details |
| 8 | Implementation and compute — starting plan accepted | Python/PyTorch first core, reference reuse, incremental integration, local CPU/MPS profiling and data/caches accepted in D50. Exact environment, operator support, storage/runtime and acceleration need remain to verify |
| 9 | Full-set rollout — accepted | Stage order and investigation/adoption policy accepted in D51; exact later-stage heads and numerical details refine during implementation |
| 10 | Linearization and data-driven modelling — final design discussion | Revisit the accepted baseline; distinguish physical model mismatch from linearization error; discuss learned residual/submodels/physical parameters, Jacobian consistency, identifiability and which additional investigations to pursue |
| 11 | Gemma 4 — future experiment order accepted | Offline annotation/teacher first, then low-frequency supplemental scene context. Detailed execution deferred; no initial-pipeline dependency |

Items 1–9 are accepted at architectural/policy/starting-configuration level. **Design discussions consolidated in [PROJECT_SNAPSHOT.md](../archive/PROJECT_SNAPSHOT.md).** Existing grey-box/analytical-filter decisions persist. Gemma has an accepted deferred experiment order; the causal post-filter error-refiner is an accepted potential post-pilot experiment and is not a prerequisite for building the baseline. Camera-pair and engineering feasibility checks remain implementation work.

### Dataset and novelty clarification

The [benchmark-data and novelty audit](./benchmark-data-and-novelty-audit.md) checks the recent papers' actual training/test data. NCLT is the most repeated named real dataset in this targeted KalmanNet-related shortlist. The currently published Song et al. preprocessing defines 22 train / 2 validation / 3 test recording days within that one dataset; this is not training on 22 separate datasets. Many other papers use generated synthetic systems only. HDVIO-family papers evaluate multiple drone datasets but use platform/experiment-specific learned models, rather than establishing one pooled universal model.

The earlier audit found close related methods, including HDVIO/HDVIO2.0 and analytical-UKF vehicle learning. The user subsequently clarified that novelty is not required. Retain those papers as technical/evaluation references; no further novelty search is needed to justify continuing.

**User decision: NCLT is locked first.** Keep VID indoor and the 2024 racing-car data only as optional future investigations. Five planar outputs, internal attitude, propagation, error-state EKF/MSCKF backend and bounded local geometry, learning roles, visual route, exclusions and dropout diagnostics are agreed. Exact augmented-state/neural/implementation contracts remain open. See the [sensor and state specification](./nclt-sensors-and-state.md).

## Reference / ground-truth questions

### Why another estimator can be valuable

A useful deployed estimator may use cheaper or lighter sensors, fewer external services, a causal low-latency algorithm, or a different sensor subset. A laboratory reference can use external instruments or future measurements unavailable in deployment. A new estimator can also address states such as force or slip that the reference navigation package does not directly deliver.

Identical sensor categories do not imply identical physical instruments or data streams. Conversely, if a reference and the proposed estimator use exactly the same recordings, dependence must be acknowledged. An offline solution may have an advantage through future data, but its output is not automatically independent truth. A learned model additionally brings prior information acquired during training.

### Concrete reference arrangements

| Example | What the reference actually contributes | Meaningful comparison |
|---|---|---|
| **UrbanNav Hong Kong** | Documentation lists an Xsens IMU and consumer GNSS receivers separately from SPAN-CPT RTK GNSS/INS | Estimate from a declared low-cost sensor subset and score against the separate reference; inspect reference quality in outages and urban canyons |
| **NTU VIRAL** | A Leica total station tracks a physical prism externally | Score the estimator's lever-arm-corrected prism position; this is external optical metrology, not another solution derived only from the onboard navigation streams |
| **Boreas** | Applanix, wheel encoder, RTX corrections and postprocessed batch/smoothing solution | A causal camera/LiDAR/radar/raw-IMU estimator can have different information; raw Applanix IMU sharing creates reference correlation, while processed IMU products can leak smoothing information |
| **Mocap datasets** | External optical observations of markers and calibrated reconstruction | Usually clearer separation from onboard sensing; still subject to calibration, timing, occlusion, frame and derivative-estimation errors |

[UrbanNav sensor list](https://raw.githubusercontent.com/IPNL-POLYU/UrbanNavDataset/master/README.md), [NTU sensor/reference documentation](https://ntu-aris.github.io/ntu_viral_dataset/sensors_and_usage.html), [Boreas reference/input provenance](https://raw.githubusercontent.com/utiasASRL/pyboreas/master/DATA_REFERENCE.md).

Outdoor data can therefore have an external optical reference or a separate, better-instrumented navigation reference. When independent instruments are impractical, a well-characterized postprocessed solution can be useful, provided claims are limited by its uncertainty and correlations. Simulation supplies exact simulator state, but cannot replace evidence about real-world model error.

### What a reference supports

The measured comparison is

```text
estimate - reference = (estimate - true_state) - (reference - true_state)
```

Agreement with a reference is not identical to error against physical truth. Shared errors can make agreement misleading. Position truth alone cannot validate orientation or an external force independently; a differentiated pose trajectory is a derived velocity reference, not an independent velocity instrument.

For selection, document measured quantities, coverage, reference uncertainty, input/reference shared information, and the size of the claimed gain. An apparent gain smaller than poorly characterized reference error needs additional evidence. Held-out sensors or external metrology can help, but consistency on held-out measurements still must be interpreted with their own noise.

## EuRoC accuracy clarification

The user clarified that their reported meter-scale results came from a camera-free, effectively dead-reckoning estimator over the complete trajectory. These results address a different information setup from camera-plus-IMU visual-inertial/SLAM results. Further debugging of that earlier project is not requested.

The original ORB-SLAM3 paper reports an average stereo-inertial accuracy of 3.6 cm. Its Table II gives RMS ATE of 0.036 m on MH01 and 0.033 m on MH02 for stereo-inertial ORB-SLAM3; the same table gives 0.166 m and 0.152 m for stereo-inertial VINS-Fusion. These are published trajectory-evaluation results, not reproduced here. ORB-SLAM3 is a SLAM system with place recognition/loop closure and map refinement. Its evaluation aligns trajectories with SE(3) for these sensor configurations and reports median outcomes over repeated runs. It is not equivalent to raw absolute-coordinate error, final dead-reckoning error, or strictly causal filter accuracy. [Original paper and Table II](https://arxiv.org/html/2007.11898).

Future comparisons must match sensors, causality, alignment and metrics. This is a prospective evaluation requirement, not a request to diagnose the old experiment. The user remains open to EuRoC where it fits the chosen physical estimation problem.

## Current decision state and next discussion

- Agreed objective: improved state estimation, or comparable performance with measurable compute, causality/history, latency or sensor advantages. Forces and other physical quantities are auxiliary candidates.
- Agreed first dataset/platform: **NCLT Segway**, for development, training and evaluation. VID indoor and the IROS 2024 racing-car data are optional future work.
- Agreed process: follow a promising approach and evaluate it; novelty/publication is not required.
- Agreed preference: retain physics and learn residual/bias means together with uncertainty adaptation. Backend, modular learning organization and pilot interfaces are selected; detailed mathematical and numerical choices remain.
- Agreed architecture: five planar navigation outputs, tilt-aware internal attitude, IMU/FOG-assisted propagation, wheel/GNSS/visual updates and three learning roles; exclusions, camera configurability and removal diagnostics persist.
- Backend, pilot ownership, GRU/timing/memory, startup/recovery, baseline mathematics, initial numerical/visual configuration and training goals locked; evaluation/comparisons, implementation/compute and the initial training protocol are accepted; rollout order is accepted; current discussion is [model learning and Gemma 4](./model-learning-and-gemma4.md). The small initial camera subset still needs a pair-selection audit; no depth-optimal pair has been measured. Lateral no-slip is planned within the project; linearization/data-driven modelling is the final dedicated design discussion. One small sensor archive and corrected camera calibration ZIP have been audited; no estimator has been implemented or evaluated.
- Hardware: MacBook Air M5 / 16 GB RAM; small local learning experiments are a working feasibility assessment, not a measured performance claim.
- Not yet agreed: numerical/network settings, later-stage full-target parameterization, enhanced linearization/data-driven modelling, implementation/codebase, session split/protocol, numerical targets, benchmark methods and training budget. Internal 3D state, reported planar state, baseline model family and exact pilot outputs are settled; detailed sensor/noise derivations are implementation work.
- Prior EuRoC clarification is resolved; no debugging work is requested.

## Sources added for this decision stage

- [VID dataset paper](https://arxiv.org/abs/2103.11152) and [official sequence/channel documentation](https://github.com/ZJU-FAST-Lab/VID-Dataset).
- [HDVIO, RSS 2023](https://arxiv.org/abs/2306.11429).
- [HDVIO2.0, T-RO 2025](https://rpg.ifi.uzh.ch/docs/tro25_Cioffi.pdf) and [public implementation](https://github.com/uzh-rpg/hdvio2.0).
- [NeuroBEM data provenance and physical parameters](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html).
- [TartanDrive 2.0 official site](https://theairlab.org/TartanDrive2/) and [2024 paper](https://arxiv.org/abs/2402.01913).
- [ORB-SLAM3 original paper](https://arxiv.org/html/2007.11898).
- [UrbanNav official sensor list](https://raw.githubusercontent.com/IPNL-POLYU/UrbanNavDataset/master/README.md).
- [NTU VIRAL external tracker documentation](https://ntu-aris.github.io/ntu_viral_dataset/sensors_and_usage.html).
- [Boreas reference construction and raw/processed IMU distinction](https://raw.githubusercontent.com/utiasASRL/pyboreas/master/DATA_REFERENCE.md).
- [KIT Multi-Surface Driving Maneuvers institutional record](https://publikationen.bibliothek.kit.edu/1000192628) and [archive/signal metadata](https://www.radar-service.eu/radar/en/dataset/44a91t97pmnha1k9).
- [Learning dynamics models for velocity estimation in autonomous racing, IROS 2024](https://arxiv.org/html/2408.15610).
- [TU/e real F1tenth driving data](https://zenodo.org/records/12536536) and [README](https://zenodo.org/records/12536536/files/ReadMe.md?download=1).
