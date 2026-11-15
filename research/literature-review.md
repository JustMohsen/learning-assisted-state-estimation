# Learning-assisted state estimation: literature and dataset review

Initial research: 12 November 2026. Updated: 13 November 2026. **NCLT is selected as the first development, training and evaluation dataset.** VID indoor and the IROS 2024 racing-car data are optional future investigations. Exact estimator/data protocol remains open; novelty is not a project requirement.

Current decision process: [project goals and decision record](./project-decisions.md). Active scope is indoor drones and road vehicles, with state-estimation accuracy or deployment efficiency as the scientific target. Learned residuals and uncertainty are candidate tools. See the [modeling/vision/compute note](./modeling-and-compute-options.md) for the current discussion. Other application branches below remain background research. Earlier dataset recommendations address general navigation/fusion and are not a final selection.

Latest architecture decisions and targeted reading: start with frozen **SuperPoint + LightGlue** and retain analytical Kalman gain/correction. The [architecture note](./architecture-options.md) now records original feature/matcher benchmark numbers, stereo geometry, online scale/bias calibration, and flowcharts for Backprop KF, Recurrent Kalman Networks and KalmanNet. Other visual methods and fine-tuning are optional; model-matrix/linearization corrections are deferred for the next discussion.

The user clarified that the requested filter discussion concerns **KalmanNet variants and subsequent models**. See the focused [variant review and flowcharts](./kalmannet-variants.md), including the distinction between original/Split velocity-only examples and later actual navigation-fusion implementations. The latter do not imply that the published learned-gain architecture has been selected for this project.

The [benchmark-data and prior-work audit](./benchmark-data-and-novelty-audit.md) records training/test protocols paper by paper. NCLT is the most repeated real dataset in this targeted learned-filter family; that is different from a field-wide dataset ranking. The user has selected NCLT and explicitly stated that novelty is unnecessary. The [NCLT modeling options](./nclt-modeling-options.md) are the current design discussion.

## 1. Main assessment

The proposed project has a substantial literature behind it. The most useful framing is **causal estimation of a physical navigation or dynamics state from complementary sensors, with learning used either throughout the estimator or in selected probabilistic/model components**.

The initial broad review identified MILUV, UTIL and UrbanNav as candidates for different navigation/fusion problems, and NeuroBEM for aerodynamic modeling. These recommendations are historical: **NCLT is now selected first**. SHIRT and other domains remain background reading rather than active dataset choices.

These are project-fit judgments, not a ranking of benchmark accuracy. A worthwhile research question is whether a learned estimator remains accurate **and** reports useful uncertainty when measurement quality, motion, geometry, or the platform changes. Merely adding a network to a Kalman filter is an established design pattern.

## 2. Search scope and strength of evidence

The search covered neural filtering, differentiable Bayesian filters, covariance/gain learning, learned observation biases, residual dynamics, physics-informed observers, visual/inertial fusion, UWB navigation, GNSS/INS, vehicle sideslip and articulated trucks, underwater navigation, legged robots, and spacecraft relative navigation. Foundational work from 2015 onward was connected to selected 2025–2026 developments.

Sources were prioritized in this order: original papers; official dataset documentation; author-maintained implementations; institutional publication records. Search aggregators were used for discovery, rather than as authority for benchmark results. Primary pages and selected full-text method sections were inspected, including the original KalmanNet navigation experiment, UTIL baseline source, MILUV limitations and benchmarks, Boreas input/reference provenance, and the 2026 Unscented KalmanNet flight experiment.

This is a broad scoping review, not a PRISMA-style systematic review. “Common” below means recurring across the reviewed literature, not a measured percentage of all publications. The 13 November update adds a dated bibliometric snapshot from the OpenAlex API for selected dataset papers; it does not enumerate every publication that actually benchmarks on each dataset. Linked implementations were inspected selectively and have not been installed or reproduced. A public repository does not establish that a complete training dataset or maintained environment is available.

## 3. State estimation, odometry, and observability

A typical 3D navigation state is

```text
state[t] = (position[t], velocity[t], orientation[t],
            accel_bias[t], gyro_bias[t])
```

with position, velocity, orientation, accelerometer bias, and gyroscope bias. Other problems add wheel slip, sensor time offsets, clock states, aerodynamic parameters, or external forces.

The corresponding inference target is a filtering distribution such as

```text
p(state[t] | measurements[1:t], inputs[1:t])
```

where measurements available up to the current time correct a prediction. An offline smoother instead uses future measurements as well. Fixed-lag methods must state their delay.

Dead reckoning and odometry are themselves forms of state estimation. The useful distinction for this project is whether the estimator receives continuing observations that constrain drift, what physical states it maintains, and whether its uncertainty is meaningful. IMU-only displacement prediction can be valuable without satisfying the desired multi-sensor, externally corrected navigation task. Conversely, visual-inertial odometry can estimate velocity and biases while retaining global position/yaw gauge freedoms.

Anchor ranges, GNSS, and localization in a known map provide external constraints. Unmapped visual or LiDAR odometry and contact constraints provide useful corrections but do not automatically make global pose observable. A motion-capture ground-truth stream used for scoring is not an available navigation measurement.

Single-tag ranging does not directly measure full orientation. Yaw and biases may be weakly observable or unobservable for particular motions and geometries. A network can learn a motion prior, but cannot create information absent from the measurements. An estimator using multiple spatially separated tags, visual landmarks, or a heading sensor has a different observability problem.

## 4. What learning usually does

| Approach | Learned quantity | What remains analytical | Assessment for this project |
|---|---|---|---|
| Direct recurrent or attention-based estimation | State, state increments, and sometimes uncertainty from sensor history | Possibly little beyond preprocessing and geometric output constraints | Necessary comparison for the end-to-end objective; demands careful causal inputs and generalization tests |
| Learned measurement extraction | Relative pose, velocity, range correction, landmarks, contact probabilities | Filter or graph fuses the extracted information | Widespread in robotics; learned front-end performance and posterior-state performance must be separated |
| Measurement covariance learning | Time-varying observation covariance R | Measurement equation and filter/graph update | Directly addresses changing reliability; does not by itself remove systematic bias |
| Process covariance learning | Time-varying process covariance Q | Physical propagation and update | Useful for model uncertainty, but Q can absorb errors from other sources |
| Gain/update learning | Kalman gain or a bounded update residual | Some prediction and innovation structure | KalmanNet family; uncertainty needs an explicit, separately validated treatment |
| Residual/parameter learning | Correction to dynamics, measurement model, drag, tire parameters, or bias evolution | Known kinematics/dynamics and inference | Particularly aligned with the proposed modeling-error and disturbance interests |
| Learned switching/fusion | Mode probabilities, measurement weights, or expert selection | Multiple physical models or estimators | Natural extension of interacting multiple-model filtering; include a classical switching baseline |
| Physics-informed training | Dynamics residuals, geometric constraints, conservation relations, or observer constraints | Depends on architecture | Promising, but a physics loss is different from retaining a physical filter recursion |

These categories overlap. **End-to-end training is a training strategy, not a synonym for a black-box architecture.** A neural encoder plus differentiable EKF can be trained from the final state error through the entire estimation pipeline. This makes it possible to compare a direct sequence network with a structured, physics-guided estimator under the same training objective.

The surveys [R01–R02], differentiable-filter studies [R03–R05], and neural Kalman literature [R06–R10] provide complementary entry points.

### Covariance and gain learning

Distinguish Q (process uncertainty), R (measurement uncertainty), P (posterior-state uncertainty), and K (correction gain). Learning one is not equivalent to learning all four. A matrix-valued network output requires a valid covariance parameterization, for example L L^T plus a positive diagonal floor. Positive definiteness establishes algebraic validity, not statistical calibration.

A network can hide a biased measurement behind a large R. It is often more interpretable to learn a measurement mean/bias correction and uncertainty separately. Joint Q/R learning from innovations alone can be poorly identifiable. Calibration, regularization, and deliberate ablations are therefore part of the scientific question.

### Residual learning and environmental perturbations

A representative model is

```text
state[t+1] = physical_model(state[t], inputs[t])
             + residual_network(state[t], inputs[t], causal_history[t])
             + process_noise[t]
```

On rotations, the correction should be composed on the appropriate manifold rather than added indiscriminately to orientation coordinates. An analogous residual can be placed in the measurement equation.

This is useful when a nominal model misses repeatable effects. However, “learned residual acceleration” is not automatically “estimated wind”: wind, thrust error, attitude error, mass error, and accelerometer bias can produce similar discrepancies. Force or aerodynamic claims need appropriate actuator/control data and additional assumptions or labels. Also distinguish IMU-driven kinematic propagation from rotor-driven rigid-body dynamics; adding an aerodynamic model to an IMU pipeline can double-count acceleration effects unless its role is defined carefully.

### Physics-informed learning

Three meanings recur: retaining physical equations inside an estimator; learning a residual around a physical model; and adding differential-equation residuals to the training loss, as in PINNs. They should be evaluated separately. An offline network fitted to an entire trajectory can exploit future observations, and is not automatically a deployable causal observer.

## 5. Application landscape and research groups

| Application/group or representative line | States and geometry | Sensors/data used | Research approach |
|---|---|---|---|
| Neural filtering: Revach, Shlezinger, Eldar and collaborators | General nonlinear states; simplified navigation examples | Synthetic state-space systems; NCLT in original KalmanNet | Learn the correction gain while retaining state-space structure [R06] |
| UWB aerial localization: Zhao, Goudar, Panerati, Schoellig | Drone position/pose in 3D | UWB and inertial sensing; related work and UTIL | Correct range/TDOA bias and reject outliers; public localization data [R13–R14, D02] |
| Multi-UAV fusion: Shalaby, Dahdah, Ahmed, Cossette, Le Ny, Forbes | 3D single- and multi-robot states | MILUV | Classical fusion benchmarks plus low-level radio-learning examples [D01] |
| Continuous-time fusion: Li, Cao, Hanebeck | Continuous 3D motion, with asynchronous sensors | UWB/IMU; UTIL and author-collected data | Sliding-window quaternion splines and calibration; a strong classical comparator [R15] |
| Urban GNSS: Li, Hsu and collaborators at PolyU | Ground-vehicle navigation; full 3D model possible, often horizontal error reporting | UrbanNav GNSS/IMU | CNN-guided GNSS reliability and adaptive factor graph fusion [R16] |
| Differentiable robotics: Haarnoja and collaborators; Kloss, Martius, Bohg; Lee and collaborators | Physical object/robot state; several controlled or planar tasks | Images, tactile/force information, simulated and real manipulation | Train model/noise components through recursive inference [R03–R05] |
| Quadrotor dynamics: Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza | 3D flight dynamics, forces and torques | NeuroBEM motion capture and onboard/motor data | First-principles aerodynamics plus learned residual dynamics [R17, D10] |
| Passenger-vehicle dynamics: Bertipaglia, Alirezaei, Happee, Shyrokau | Primarily planar lateral dynamics and sideslip | Experimental vehicle measurements | CNN informed by UKF/single-track vehicle model [R18] |
| Articulated trucks: Ewering, Ziaukas, Ehlers, Seel | Articulation, steering, tire-force-related states | Full-scale truck/semitrailer experiments | ANN-aided EKF; test generalization across payloads [R19] |
| Spacecraft relative navigation: Park, D’Amico; Cassinis and collaborators | Relative translation and attitude; temporal filtering may infer rates | SPEED+/SHIRT images | Neural visual measurements/uncertainty combined with adaptive UKF [R20–R21, D12–D13] |
| Underwater navigation: Cohen, Klein | 3D navigation | Real AUV IMU and Doppler velocity log | Transformer adapts process covariance in an EKF [R22] |
| Legged robots: Lin, Zhang, Yu, Ghaffari | Base pose/velocity and contact-related states | IMU and proprioceptive joint information | Learn contact events to assist invariant filtering [R23] |

Road navigation, visual/inertial robotics, learned inertial motion, and vehicle dynamics have particularly extensive literatures. Spacecraft and marine work is substantive, but readily reusable multi-sensor data with independent full-state truth is less straightforward to obtain. Beyond mobile systems, battery state of charge/health, electrical power-system state, and process/structural estimation are also major ML state-estimation applications; they answer a different physical question from vehicle navigation.

Ground vehicles often use planar models for lateral dynamics, even when the recorded reference pose is 3D. Drones, spacecraft, and underwater vehicles naturally motivate 3D translation and attitude. A dataset's 3D trajectory does not mean a particular paper estimated the full 3D navigation state.

## 6. Dataset comparison

“Reference” is used deliberately: many published ground-truth trajectories are themselves estimates. A dataset can provide measured pose without independent velocity, bias, force, or covariance truth. Dataset contents and baseline availability below are source-checked; reproduction remains future work.

**How to read the new columns.** Publication dates distinguish an earlier release/preprint or conference from a later journal paper where relevant. Citation counts are **OpenAlex indexed citing works, queried on 13 November 2026**, for the specific linked records. They include references made without running the dataset and may include preprints and other publication types. They are not Google Scholar counts, unique dataset-user totals, or verified benchmark-paper counts. Different paper versions are reported separately and are not added together. The selected records, DOI mappings, and returned counts are preserved in the [citation snapshot](./dataset-citation-snapshot.json).

Access and benchmarking effort are qualitative assessments from documentation, not measured installation times or successful fresh executions. **Low / medium / high describe implementation effort; motion/scene difficulty is a separate column.** A slow ground platform can still have difficult visual scenes, and a controlled drone dataset can still require full 3D attitude estimation.

| Dataset | Publication / release year | Indexed citing works (OpenAlex, 2026-11-13) | Platform / dimension | Available measurements relevant to fusion | Reference and coverage | Access / ease of use | How to benchmark / estimated effort | Motion / scene difficulty | Classical comparison / adoption evidence and project fit |
|---|---|---|---|---|---|---|---|---|---|
| **MILUV** [D01] | 2025 preprint; 2026 IJRR | [5](https://openalex.org/W7120228190) | Indoor drones; 3D; up to three robots | UWB, cameras, two IMUs per robot, height, magnetometer; CIR in selected trials | Vicon pose | **Medium.** Public Figshare data; Python loader and examples. Select experiments; dependency setup needs checking | **Medium** for single-robot UWB/IMU; higher for visual or cooperative fusion. Reuse devkit EKFs; score pose against Vicon | Controlled room; variable 3D pace, occlusions, and multi-robot geometry | New adoption base; SE(3)/SE_2(3) EKFs and VINS-Fusion supplied. Strong sensor-fusion fit, limited independent comparison history |
| **UTIL** [D02] | 2022 preprint; 2024 IJRR | [63](https://openalex.org/W4391543839) | Indoor drone; 3D | UWB TDOA, IMU, optical flow, ToF height, barometer | Vicon pose; surveyed anchors | **Low–medium.** Public CSV and ROS bags; Python/MATLAB parsing; no camera front end required for UWB/IMU | **Low–medium.** Start with supplied ESKF and position RMSE; add attitude/velocity evaluation as justified. SFUISE provides a stronger comparator | Slow controlled flights on average; select LOS first. NLOS and dynamic obstacles add difficulty | Smaller specialist ecosystem; official page identifies follow-up estimation papers. Best manageable first fusion experiment |
| **UrbanNav** [D03] | 2021 conference; 2023 journal | [147](https://openalex.org/W3205114100) conference; [106](https://openalex.org/W4383220232) journal; overlapping citing sets possible | Road vehicle; 3D navigation | Raw GNSS, IMU, cameras, LiDAR; subset-dependent | SPAN-CPT GNSS/INS reference | **Medium.** Per-subset downloads; separate RINEX, IMU, truth and bag files; some full sensor subsets are tens of GB | **Medium** for position-solution/IMU fusion; **high** for raw tightly coupled GNSS. Match times/frames; compare positioning RMSE and availability on the same routes | Less aerial rotation; urban multipath, dynamic traffic and outages can be severe | Established within GNSS navigation; GraphGNSSLib and published baselines. Strong road alternative, not automatically an easier estimation problem |
| **NTU VIRAL** [D04] | 2021 online; 2022 IJRR issue | [215](https://openalex.org/W3214208961) | Aerial; 3D | Stereo, two LiDARs, IMUs, UWB | Leica prism **position**, not independent attitude | **Medium.** Official Hugging Face mirror, bags, calibration and truth CSV; multi-GB sequences and custom UWB messages | **Medium–high.** Dataset-specific baseline configs/evaluator exist; compensate prism lever arm. Position evaluation is clearer than full-state validation | Variable indoor/outdoor 3D flight; later sequences add scene challenges | Broad public method/config catalogue. Its 63 method/implementation entries are **not** 63 verified benchmark publications |
| **EuRoC MAV** [D05] | 2016 IJRR | [2,172](https://openalex.org/W2396274919) | Indoor drone; 3D | Stereo cameras and IMU | Vicon or Leica, plus processed state references | **Low–medium.** Public ASL files/bags, calibration, many ready-made configurations | **Low–medium** with an existing VIO implementation; **high** for building a visual front end from scratch. Standard trajectory evaluation and many published comparisons | Explicit easy/medium/difficult sequences. Machine Hall 01/02 offer gentler starting points than difficult Vicon-room runs | Very mature VIO ecosystem: OpenVINS, VINS and other established pipelines. Strong comparison dataset; does not require dense 3D mapping |
| **TUM VI** [D06] | 2018 IROS | [523](https://openalex.org/W2797395780) | Handheld; 3D | Stereo fisheye cameras and IMU | Full mocap in room sequences; many long routes only start/end | **Low–medium.** Public downloads/calibration and multiple image resolutions; established VIO configs | **Medium.** Use room sequences for dense pose scoring. Long sequences often need endpoint drift evaluation rather than dense-state RMSE | Handheld motion, fisheye calibration and initialization; some long routes are very challenging | Mature VIO comparison ecosystem. A different motion regime, but not a guaranteed simpler task |
| **Boreas** [D07] | 2022 preprint; 2023 IJRR | [189](https://openalex.org/W4327692895) | Road vehicle; 3D, repeated seasons | Camera, LiDAR, radar, raw IMU and wheel data | Postprocessed Applanix pose/velocity | **Medium.** Public AWS, no account required with unsigned requests; pip-installable pyboreas. Download selected sensors/sequences | **Medium–high.** Devkit evaluation and public odometry/localization leaderboards; asynchronous streams and large data. Use raw IMU deliberately | Road motion; weather, seasonal changes and map aging are major challenges | Active odometry/localization ecosystem and official evaluation tools. Attractive established outdoor option with a larger processing footprint |
| **NCLT** [D08] | 2015 online; 2016 IJRR issue | [590](https://openalex.org/W2335829695) | Segway ground robot; 3D reference, often planar treatment | Cameras, LiDAR, IMU, wheel odometry, consumer/RTK GPS, fiber-optic gyro | Offline multi-session LiDAR/RTK/odometry SLAM graph with odometry-interpolated dense reference | **Low–medium** for sensor-only archives; images/LiDAR much larger. Direct session downloads; older custom formats | **Medium.** Public navigation-fusion KalmanNet code and session splits exist; define sensor subset, frames, reference and timing consistently | Mostly ground motion; long sequences and seasonal changes; not a passenger-car actuator dataset | Most repeated named real dataset in our reviewed KalmanNet family; six named uses established in the [targeted audit](./benchmark-data-and-novelty-audit.md). This count is distinct from citations and does not imply identical protocols |
| **M2DGR** [D09] | 2021 online/preprint; 2022 RA-L issue | [331](https://openalex.org/W4285512293) | Ground robot; 3D pose | Multiple visual modalities, LiDAR, IMU, GNSS | RTK/INS, Leica or mocap by sequence | **Medium.** Public sequence downloads, sensor configs and extraction/evaluation guides; full suite exceeds 1 TB | **Medium–high.** M2DGR-Benchmark and public configurations reduce effort; select a consistent reference modality and exact dataset version | Mild platform motion can coexist with darkness, lifts and indoor/outdoor transitions | Many public evaluated systems; counts for M2DGR do not represent all users of its later extensions. Strong ground-robot option |
| **NeuroBEM** [D10] | 2021 RSS | [166](https://openalex.org/W3177010373), model-paper citations | Agile drone; 3D dynamics | Controller/onboard data, motor speeds, voltage, motion capture | Pose plus processed derivative quantities | **Low** for processed CSV modeling; public archive is about 627 MB compressed | **Low–medium** for reproducing force/dynamics prediction; **high** for defining a new externally aided state-estimation benchmark | Deliberately agile flight; not a gentler-motion replacement for EuRoC | Established dynamics-model reference; citation count is not navigation benchmark adoption |
| **Blackbird** [D11] | 2018 ISER/preprint; 2020 IJRR | [62](https://openalex.org/W2895010059) proceedings; [44](https://openalex.org/W3011186507) journal; do not sum | Aggressive drone flight; 3D | Real IMU/rotor data with virtual camera renderings | Motion capture | **Medium–high.** ROS/Docker tooling; selectively download from a very large collection. Maintainers flag unavailable pre-rendered sequences | **Medium–high.** Match flight, speed and rendered scene; sensor messages and perception baselines require setup | Deliberately aggressive, with selectable speed variants | Smaller adoption base and current access friction; not the first choice for reducing implementation risk |
| **SPEED+** [D12] | 2021 dataset release; 2022 IEEE Aerospace paper | [195](https://openalex.org/W3202897977), dataset-paper citations | Spacecraft; relative 6-DoF pose | Synthetic and laboratory monocular imagery | Renderer/testbed pose labels | **Low–medium** for data access: public Stanford repository, images and labels | **Medium** for visual pose scoring; **high** to construct a new temporal/multisensor estimator. Dataset alone is not a navigation time series | Illumination, visual ambiguity and synthetic-to-real gap | Established spacecraft visual-pose benchmark. Strong image-pose comparison, different from sequential fusion |
| **SHIRT** [D13] | 2022 release/preprint; 2023 associated method paper | [81](https://openalex.org/W4383909160), **associated method-paper proxy** | Spacecraft rendezvous; 3D relative-pose sequences | Synthetic and hardware-in-the-loop images | Trajectory/testbed labels and metadata | **Low** for data access: one approximately 3.01 GB ZIP, images and JSON metadata | **Medium–high.** Follow adaptive neural/UKF literature and pose metrics; dynamics/frame conventions require care. Few distinct trajectories | Limited trajectories; tumbling, image-domain gap and visual pose ambiguity | Specialist temporal spacecraft literature; method citations are not verified SHIRT-use counts |
| **Newer College** [D14] | IROS 2020; online proceedings 2021 | [266](https://openalex.org/W3012175130) | Handheld; 3D | Camera, LiDAR, IMU | LiDAR registration against surveyed map | **Medium.** Official download/mirror options; bags/calibration/reference; data volume and variant choice matter | **Medium–high.** Mature LIO/VIO tooling can be adapted; document original versus later extensions and trajectory evaluation | Walking-speed original routes; added aggressive-motion/lighting sequences | Established LiDAR/inertial benchmark. Reference shares LiDAR information; not automatically simpler for raw-vision learning |
| **AQUALOC** [D15] | 2019 IJRR | [144](https://openalex.org/W2981635988) | Underwater ROV; 3D | Monocular camera, IMU, pressure | Dense independent full-state truth not established by reviewed documentation | **Medium.** Public landing page with raw/bag downloads; older software stack | **High** for this proposed supervised full-state task until reference coverage/provenance and scoring are audited | Underwater lighting, contrast and visual degradation | Recognized underwater localization data; less straightforward supervised-estimation benchmark |
| **KITTI** [D16] | 2012 benchmark; 2013 IJRR data paper | [10,205](https://openalex.org/W2115579991), **2013 broad dataset paper only** | Road vehicle; 3D pose, largely road-constrained motion | Stereo, LiDAR and GPS/IMU products; raw and odometry packages differ | OXTS GPS/INS trajectory products | **Low–medium.** Registration for official downloads; widespread loaders/configs; select sensor packages | **Low–medium** for established odometry scoring: official 100–800 m segment metrics and devkit. **Medium–high** to define fair full-state fusion. Public test submission has publication restrictions | Generally less aerial rotation; traffic, scale and longer trajectories still challenging | Very broad CV adoption; count includes detection/depth/etc., so is not a navigation-paper count. Good conventional comparison if task matches |

### Adoption counts and benchmarking evidence

The numeric column answers **how many indexed works reference the selected paper**, not how many groups actually ran its data. A defensible complete benchmark-use count would require inspecting and deduplicating citing works, including papers that cite a different dataset version. It has not been established here; a missing total must not be read as zero. GitHub stars, method ports and leaderboard entries are also different quantities.

Several version issues materially affect interpretation:

- **UrbanNav:** the 2021 conference and 2023 journal papers have different records and potentially overlapping citing works. Report 147 and 106 separately, not 253 users.
- **Blackbird:** likewise, report 62 proceedings-record and 44 journal-record citations separately. OpenAlex dates the proceedings volume to 2020, while the underlying event and author citation are ISER 2018.
- **MILUV/UTIL:** querying their preprint DOI aliases resolved to the same selected journal records in this audit. They were counted once each.
- **NeuroBEM/SHIRT:** the linked records concern a model or filtering method as well as the dataset. Citation counts are particularly imperfect proxies for dataset reuse.
- **KITTI:** 10,205 refers only to the 2013 data paper, excludes any union with the 2012 benchmark paper, and includes computer-vision tasks outside state estimation.
- **Age:** low counts for MILUV reflect its recent publication as well as its adoption. Citation counts alone cannot establish dataset quality or likely implementation difficulty.

Public evidence for benchmarking is therefore listed independently of the counts: the [NTU VIRAL catalogue](https://ntu-aris.github.io/ntu_viral_dataset/) separates method implementations and tools; [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) lists tested systems and an evaluation framework; [pyboreas](https://github.com/utiasASRL/pyboreas) provides official odometry/localization evaluation support; and [OpenVINS](https://docs.openvins.com/gs-datasets.html) documents EuRoC and TUM VI configurations. These establish practical comparison options without pretending to be complete paper censuses.

### Choosing a manageable first benchmark

**EuRoC is not uniformly aggressive, and using it does not require dense 3D mapping.** It includes easy, medium and difficult sequences; Machine Hall 01/02 are explicit easy entries. A sparse visual-inertial filter can estimate motion without reconstructing a dense scene. The [OpenVINS dataset guide](https://docs.openvins.com/gs-datasets.html) actually recommends EuRoC as a first dataset and provides default configurations, while noting initialization and reference issues. It also flags an orientation-reference issue for V1_01_easy, so that sequence should not be selected casually for independent attitude evaluation.

For the stated concern, choose among different kinds of simplicity:

1. **Simplest measurement-processing path for the original fusion goal: UTIL, initially LOS.** UWB and IMU can be processed numerically without building a camera or point-cloud front end. It remains a 3D estimator in a small indoor arena; simplicity comes from the sensors and controlled motion, not from avoiding all spatial constraints.
2. **Most established VIO comparison path: selected easy EuRoC sequences.** The difficult part is the visual front end; ready-made configurations and comparison papers substantially reduce integration uncertainty. It can be a secondary benchmark even if it is not the first training dataset.
3. **Ground motion with a compact sensor subset: NCLT GPS/wheel/IMU.** Older formats and reference correlations need attention, but raw image/LiDAR processing can be omitted. UrbanNav is more direct for GNSS research, with substantially harsher measurement failures.
4. **Richer drone fusion with room to grow: MILUV.** Start with one robot and one measurement model. It offers more modalities and cooperative extensions, but fewer independent published comparisons and more dependency/geometry choices.
5. **Established ground perception/fusion: M2DGR or Boreas.** These trade aerial motion for larger data and perception/environment challenges. Select a modest sequence before committing to full-sensor processing.

None of these assessments implies that the full dataset is easy or that a newer dataset is easier than an older one. A useful initial progression is **simple sensor subset + benign sequences + classical baseline**, followed by the difficult conditions the learned component is meant to address.

### Dataset details that change the recommendation

**MILUV** was released as a 2025 preprint and published in IJRR in 2026. The paper describes 36 experiments and 217 cumulative flight minutes. Its CIR trials have much lower UWB rates, around 5 Hz, than the higher-rate range trials. Its devkit can derive smoothed IMU biases from motion capture; these are derived labels, not independent bias truth, and must not become evaluation-time inputs in a realistic bias-estimation experiment. [D01]

**UTIL** offers about 150 flight minutes, several anchor geometries, and obstruction scenarios. The public baseline code estimates position, velocity, and attitude error, rather than a 15-state bias-augmented navigation state. Adding bias states would be a deliberate extension. Timing documentation reports approximately 10 ms ground-station latency for the logged motion-capture stream. Signal-quality identification experiments and flight experiments must be checked separately before assuming the same ML features exist in both. [D02]

**UrbanNav** makes raw GNSS observations available, which is especially useful for learning measurement reliability. Use a consumer measurement stream as estimator input and preserve the reference system for scoring. Reference quality can still degrade in difficult environments; check subset flags, intervals, and frame conventions. [D03]

**NTU VIRAL** explicitly states that it lacks orientation ground truth. A filtered onboard orientation output is not an independent attitude reference. Also evaluate the tracked prism with the correct body-to-prism lever arm. [D04]

**Boreas** documents a particularly important distinction: its convenient processed IMU stream is extracted from the postprocessed Applanix solution; a separate raw IMU stream exists. Its reference is obtained using GNSS, IMU, and wheel measurements with smoothing. Choose inputs deliberately and document shared information. [D07]

**NeuroBEM** includes filtered derivatives, some using both motion capture and onboard measurements. Those are useful modeling targets, but must not silently become causal sensor inputs. It is a dynamics dataset, not an immediately interchangeable UWB/GNSS navigation benchmark. [D10]

For all datasets, separate calibration performed in advance from corrections computed from a held-out trajectory's truth. Fixed calibration can be legitimate; test-trajectory oracle correction changes the problem.

## 7. Reading map and annotated literature

### Surveys and foundational methods

| ID | Reference | Why it matters / evidence boundary |
|---|---|---|
| R01 | Shlezinger et al., *Artificial Intelligence-Aided Kalman Filters: AI-Augmented Designs for Kalman-Type Algorithms*, IEEE Signal Processing Magazine, 2025. [Paper](https://arxiv.org/abs/2410.12289), [comparison code](https://github.com/ShlezingerLab/AI_Aided_KFs) | Best broad starting point for model-oriented versus task-oriented neural/KF design. Code examples do not constitute a universal real-robot benchmark. |
| R02 | Imbiriba, Straka, Duník, Closas, *Augmented Physics-Based Machine Learning for Navigation and Tracking*, IEEE TAES, 2024 issue publication; earlier online/code records use 2023. [Institutional record](https://dspace.zcu.cz/items/fed745f3-b446-4647-bc9f-e570f78fee21), [code](https://github.com/neu-spiral/APBM-TAES2023) | Direct entry to augmenting physical dynamics rather than replacing them. Includes synthetic and real tracking examples. |
| R03 | Haarnoja et al., *Backprop KF: Learning Discriminative Deterministic State Estimators*, NeurIPS, 2016. [Paper](https://papers.nips.cc/paper_files/paper/2016/hash/697e382cfd25b07a3e62275d3ee132b3-Abstract.html) | Foundational end-to-end training through a structured state estimator. Useful antidote to treating end-to-end and filtering as incompatible. |
| R04 | Kloss, Martius, Bohg, *How to Train Your Differentiable Filter*, Autonomous Robots, 2021. [Paper](https://arxiv.org/abs/2012.14313) | Practical comparison of differentiable filter choices, learning/training strategies, and uncertainty models. |
| R05 | Lee et al., *Multimodal Sensor Fusion with Differentiable Filters*, IROS, 2020. [Paper](https://arxiv.org/abs/2010.13021) | Explicit multimodal filtering in simulated/real pushing and simulated door manipulation. Several tasks are planar/manipulation, rather than aerial navigation. |
| R06 | Revach et al., *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics*, IEEE TSP, 2022. [Paper](https://arxiv.org/abs/2107.10043) | Core learned-gain architecture. Its NCLT experiment uses velocity-only odometry with position/velocity modeled separately per axis; it is not a demonstrated full multisensor 6-DoF navigation estimator. |
| R07 | Choi et al., *Split-KalmanNet: A Robust Model-Based Deep Learning Approach for State Estimation*, IEEE TVT, 2023. [DOI](https://doi.org/10.1109/TVT.2023.3270353) | Splits recurrent covariance-related learning while using the measurement Jacobian to construct the gain. Useful comparison to direct gain regression. |
| R08 | Dahan et al., *Bayesian KalmanNet: Quantifying Uncertainty in Deep Learning Augmented Kalman Filter*, IEEE TSP, 2025. [Record and DOI](https://dspace.zcu.cz/items/62e094cd-6549-4727-b3e7-9e7605ce75ac) | Addresses uncertainty in neural-aided filtering. Calibration must still be assessed on the intended application. |
| R09 | *Recursive KalmanNet: Deep Learning-Augmented Kalman Filtering for State Estimation with Consistent Uncertainty Quantification*, EUSIPCO, 2025. [Paper](https://eusipco2025.org/wp-content/uploads/pdfs/0000885.pdf), [code](https://github.com/ixblue/RecursiveKalmanNet) | Another explicit uncertainty-oriented development; useful for designing posterior-covariance comparisons. |
| R10 | Ko, Shafieezadeh, *Unscented KalmanNet: Structure-Preserving Deep Learning with Calibrated Posterior Uncertainty under Incomplete Physics and Unknown Noise*, arXiv, August 2026. [Preprint](https://arxiv.org/abs/2608.04201) | Combines learned noise and gain corrections. Its UZH-FPV case uses IMU-integrated pseudo-velocity, not independent external position updates; reported velocity reference is derived from position. Treat as recent methodological evidence, not a demonstrated match to the requested sensor fusion. |
| R11 | Liu et al., *Deep Inference for Covariance Estimation: Learning Gaussian Noise Models for State Estimation*, ICRA, 2018. [Paper](https://people.csail.mit.edu/wrvb/files/publications/liu2018deep.pdf) | Representative measurement-uncertainty learning from raw observations. |
| R12 | Ghosh, Honoré, Chatterjee, *DANSE: Data-Driven Non-Linear State Estimation of Model-Free Process in Unsupervised Learning Setup*, IEEE TSP, 2024; 2023 preprint. [Paper](https://arxiv.org/abs/2306.03897), [code](https://github.com/saikatchatt/danse-jrnl) | Learns a prior from measurement sequences without state labels under a known linear observation setting. “Unsupervised” does not remove assumptions about measurement structure or identifiability. |

### Application-grounded hybrids

| ID | Reference | What is learned and what is estimated |
|---|---|---|
| R13 | Zhao, Goudar, Panerati, Schoellig, *Learning-based Bias Correction for Ultra-wideband Localization of Resource-constrained Mobile Robots*, IROS, 2020. [Paper](https://arxiv.org/abs/2003.09371) | Neural measurement-bias correction inside a robot-localization system; particularly close to the proposed modeling-error idea. |
| R14 | Zhao, Panerati, Schoellig, *Learning-based Bias Correction for Time Difference of Arrival Ultra-wideband Localization of Resource-constrained Mobile Robots*, RA-L, 2021. [Paper](https://arxiv.org/abs/2103.01885) | Learned TDOA bias correction combined with robust M-estimation filtering for localization under outliers. |
| R15 | Li, Cao, Hanebeck, *Continuous-Time Ultra-Wideband-Inertial Fusion*, RA-L, 2023. [Paper](https://arxiv.org/abs/2301.09033), [maintained SFUISE repository](https://github.com/ASIG-X/SFUISE) | Classical continuous-time fusion using quaternion splines; an important stronger baseline beyond a simple EKF. |
| R16 | Li et al., *Intelligent Environment-Adaptive GNSS/INS Integrated Positioning with Factor Graph Optimization*, Remote Sensing, 2024. [Paper](https://www.mdpi.com/2072-4292/16/1/181) | CNN-guided observation reliability with GNSS/INS factor-graph estimation; UrbanNav is used, with training/test areas distinguished. |
| R17 | Bauersfeld et al., *NeuroBEM: Hybrid Aerodynamic Quadrotor Model*, RSS, 2021. [Paper](https://roboticsproceedings.org/rss17/p042.html), [project](https://rpg.ifi.uzh.ch/NeuroBEM.html) | Physical aerodynamic model plus learned residual. Model prediction quality is distinct from posterior state-estimation quality. |
| R18 | Bertipaglia et al., *An Unscented Kalman Filter-Informed Neural Network for Vehicle Sideslip Angle Estimation*, IEEE TVT, 2024. [Paper](https://arxiv.org/abs/2303.05238), [published record](https://repository.tudelft.nl/record/uuid%3Aea2c9135-a0dd-43a2-8f4a-1182bd0edb43) | CNN informed by a UKF/nonlinear single-track model; experimental vehicle data. A reusable full public benchmark was not verified in this review. |
| R19 | Ewering et al., *Reliable State Estimation in a Truck-Semitrailer Combination using an Artificial Neural Network-Aided Extended Kalman Filter*, ECC, 2024. [Paper](https://arxiv.org/abs/2406.14028) | Hybrid inference of articulation, steering, and lateral tire forces from standard semitrailer sensors; evaluates changing loading conditions. A public dataset release was not established here. |
| R20 | Park, D’Amico, *Adaptive Neural-Network-Based Unscented Kalman Filter for Robust Pose Tracking of Noncooperative Spacecraft*, JGCD, 2023. [Paper](https://arxiv.org/abs/2206.03796) | Temporal spacecraft relative-pose inference with neural visual information and adaptive filtering; associated with SHIRT. |
| R21 | Cassinis et al., *Leveraging Neural Network Uncertainty in Adaptive Unscented Kalman Filter for Spacecraft Pose Estimation*, Advances in Space Research, 2023. [DOI](https://doi.org/10.1016/j.asr.2023.02.021) | Uses learned visual uncertainty in an adaptive UKF; validation on SHIRT hardware-in-the-loop imagery. |
| R22 | Cohen, Klein, *Adaptive Kalman-Informed Transformer*, Engineering Applications of Artificial Intelligence, 2025; 2024 preprint. [Paper](https://arxiv.org/abs/2401.09987) | Learns process covariance in IMU/DVL fusion using real underwater-vehicle data. DVL constrains velocity; absolute horizontal position still needs separate aiding. |
| R23 | Lin et al., *Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events*, CoRL 2021 / PMLR 2022. [Paper](https://proceedings.mlr.press/v164/lin22b.html) | Learned contact events aid an invariant filter from proprioceptive sensing. Useful hybrid design; still described as proprioceptive odometry. |

### End-to-end motion literature and a necessary counterexample

| ID | Reference | Relevance and limitation |
|---|---|---|
| R24 | Clark et al., *VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem*, AAAI, 2017. [Paper](https://arxiv.org/abs/1701.08376) | Early learned visual/inertial sequence fusion, including KITTI/EuRoC experiments; principally a relative-motion/odometry reference. |
| R25 | Chen et al., *Selective Sensor Fusion for Neural Visual-Inertial Odometry*, CVPR, 2019. [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html) | Learned modality selection/gating for imperfect sensory input; relevant to reliability-aware fusion. |
| R26 | Brossard, Barrau, Bonnabel, *AI-IMU Dead-Reckoning*, IEEE TIV, 2020. [DOI](https://doi.org/10.1109/TIV.2020.2980758) | Influential learned covariance adaptation in an invariant filter. IMU-only wheeled-vehicle dead reckoning, so a component reference rather than the final task. |
| R27 | Liu et al., *TLIO: Tight Learned Inertial Odometry*, RA-L, 2020. [Project](https://cathias.github.io/TLIO/) | Learns displacement and covariance for tightly coupled filtering. Again IMU-only inference, not external localization. |
| R28 | Mehrfard et al., *Performance Evaluation of Deep Learning-Based State Estimation: A Comparative Study of KalmanNet*, SDF, 2024. [Paper](https://arxiv.org/abs/2411.16930) | Real automotive radar evaluation finds an IMM outperforming KalmanNet. This is evidence for careful baseline choice, not a universal conclusion that neural filters fail. |
| R29 | Becker et al., *Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces*, ICML, 2019. [Paper](https://proceedings.mlr.press/v97/becker19a.html) | Uncertainty-aware filtering in a learned latent space. Latent coordinates are not automatically identifiable physical navigation states. |
| R30 | Krishnan et al., *Deep Kalman Filters*, 2015. [Preprint](https://arxiv.org/abs/1511.05121) | Foundational learned state-space modeling. Useful context, but “latent state” may refer to abstract generative variables rather than vehicle position/velocity/attitude. |

### Additional KalmanNet follow-ups from the corrected discussion

| ID | Reference | Relevance and evidence boundary |
|---|---|---|
| R31 | Ni, Revach, Shlezinger, *Adaptive KalmanNet: Data-Driven Kalman Filter with Fast Adaptation*, ICASSP, 2024; 2023 preprint. [Paper](https://arxiv.org/abs/2309.07016) | Hypernetwork/context modulation for noise-distribution changes; simulations assume a supplied/estimated noise context. Still learns K. |
| R32 | Song, Mei, Xu, Fu, Bu, *Practical Implementation of KalmanNet for Accurate Data Fusion in Integrated Navigation*, IEEE SPL, 2024. [DOI](https://doi.org/10.1109/LSP.2024.3431443), [official code](https://github.com/SongJgit/KalmanNet4SensorFusion) | NCLT GPS/wheel/IMU fusion and improved truncated-backpropagation training. Author-reported 20–40% average RMSE improvement is relative to their EKF baseline; full channel/reference/split audit remains necessary. |
| R33 | Ko, Shafieezadeh, *Cholesky-KalmanNet: Model-Based Deep Learning With Positive Definite Error Covariance Structure*, IEEE SPL, vol. 32, 2025; DOI/online lineage 2024. [Manuscript](https://par.nsf.gov/servlets/purl/10656999), [DOI](https://doi.org/10.1109/LSP.2024.3519265) | Constrains covariance-related outputs and adds uncertainty training. Positive-definite learned factors alone do not establish a conventional covariance recursion or calibrated posterior. |
| R34 | Buchnik et al., *Latent-KalmanNet: Learned Kalman Filtering for Tracking From High-Dimensional Signals*, IEEE TSP, vol. 72, 2024; 2023 preprint. [Paper](https://arxiv.org/abs/2304.07827), [published DOI](https://doi.org/10.1109/TSP.2023.3344360) | Joint observation encoding and learned filtering; high-dimensional simulated visual observations do not establish stereo VIO performance. |
| R35 | Revach et al., *RTSNet: Learning to Smooth in Partially Known State-Space Models*, IEEE TSP, 2023. [Paper](https://arxiv.org/abs/2110.04717) | Forward/backward learned smoothing; future observations make it a different inference contract from a causal filter. |
| R36 | Revach et al., *Unsupervised Learned Kalman Filtering*, EUSIPCO, 2022; 2021 preprint. [Paper](https://arxiv.org/abs/2110.09005) | Uses observation-prediction losses for label-free training/adaptation under model assumptions; hidden-state accuracy still needs independent evaluation. |
| R37 | Shen et al., *KalmanFormer: using transformer to model the Kalman Gain in Kalman Filters*, Frontiers in Neurorobotics, published January 2025 in the 2024 volume. [Paper](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1460255/full) | Transformer-based gain learning with synthetic/NCLT tests. Architecture/history/runtime suitability still requires audit for our target. |
| R38 | Li, Guo, Wang, *Enhanced KalmanNet for accurate data fusion*, Signal Processing, vol. 243, 2026. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0165168425005730) | Process-model correction plus learned K, with GPS/IMU/wheel NCLT localization experiments; close prior art for the deferred physical-model discussion. |
| R39 | Węgrzynowski et al., *Learning dynamics models for velocity estimation in autonomous racing*, IROS, 2024. [Paper](https://arxiv.org/abs/2408.15610) | Close physical-model/learning prior art: analytical UKF, physical/residual/learned tire models, trainable/state-dependent Q/R and online friction; introduced motion-capture RC-car data, with train/test and tire-transfer experiments. Global navigation pose is not the output task. |
| R40 | Cioffi, Bauersfeld, Scaramuzza, *HDVIO: Improving Localization and Disturbance Estimation with Hybrid Dynamics VIO*, RSS, 2023. [Paper](https://arxiv.org/abs/2306.11429) | Physical drone dynamics plus learned residuals within visual-inertial state estimation; evaluates NeuroBEM, Blackbird, VID and new wind experiments. Close prior art for broad architectural novelty. |
| R41 | Cioffi, Bauersfeld, Scaramuzza, *HDVIO2.0: Wind and Disturbance Estimation with Hybrid Dynamics VIO*, T-RO, 2025. [Paper](https://arxiv.org/abs/2504.00969), [code](https://github.com/uzh-rpg/hdvio2.0) | Adds translational/rotational hybrid dynamics; drone-specific training, including separate VID indoor/outdoor models. Multiple benchmark datasets do not mean one pooled training model. |

### HDVIO evidence check — 13 November 2026

OpenAlex's published DOI records currently report **16 citing works for HDVIO** (RSS 2023) and **2 for HDVIO2.0** (T-RO 2025). These are modest index-specific counts, not Google Scholar totals, independent replications or counts of papers benchmarking the methods. Preprint/published versions were not summed. The query fields and retrieval date are preserved in the [citation snapshot](./hdvio-citation-snapshot.json). [HDVIO OpenAlex record](https://openalex.org/W4385430610), [HDVIO2.0 OpenAlex record](https://openalex.org/W4413754773).

Both papers report improvements over their selected comparison systems. A concrete common example is Blackbird **Egg, 8 m/s**; these are aligned absolute trajectory errors from the HDVIO2.0 paper's Table II, not final displacement drift:

| Method | Translation ATE [m] | Rotation ATE [deg] |
|---|---:|---:|
| VIO baseline | 1.79 | 4.55 |
| VIMO | 1.57 | 3.69 |
| VID | 1.47 | 4.84 |
| HDVIO | 1.06 | 2.89 |
| HDVIO2.0 | **0.77** | **2.61** |

For that example, HDVIO2.0 reduces translation error by about **57% versus VIO** and **27% versus HDVIO**. It also demonstrates closed-loop flight. These results support investigating the approach; they do not predict NCLT gains. [HDVIO2.0, Table II and section VI](https://arxiv.org/html/2504.00969).

Original HDVIO reports up to about 33% motion-estimation improvement versus VIMO and 40% force-error improvement in its reported cases. It is not best on every sequence: Bent Dice has 0.21 m translation ATE for HDVIO versus 0.20 m for the VIO baseline. [HDVIO results](https://arxiv.org/html/2306.11429).

The recurring lesson is that method names and “real-world data” claims do not establish which measurements were actually supplied. Read the experimental state vector, observation equation, and preprocessing pipeline.

## 8. Research directions worth discussing

### A. Reliability-aware drone state estimation

Candidate data: MILUV or UTIL. Estimate a navigation state from IMU propagation and real external observations. Compare a direct causal sequence estimator against classical filtering and a differentiable hybrid.

The learned component could predict measurement bias and R from available signal/innovation history. A stronger question is whether it remains calibrated on unseen obstruction conditions, sensor geometries, and motions. Use explicit anchor coordinates or geometric features to avoid merely memorizing a room. Map/coordinate transformations should be tested consistently.

CIR-based learning is especially interesting, but the selected sequences must actually contain CIR at the required rate. UTIL TDOA and MILUV two-way-ranging observations are different measurement models; cross-dataset transfer needs a deliberate common representation and adaptation plan, not direct interchange of trained outputs.

### B. Residual-model drone estimation

Candidate data: NeuroBEM for learning a physical model; additional external sensing or an explicitly specified measurement experiment for closed-loop inference. Compare a physical model, a purely learned model, and a regularized residual model inside the same estimator.

This direction better matches aerodynamic mismatch, thrust/motor effects, and perturbations, but requires more care to distinguish system identification from filtering. It is not valid to fuse unrelated recordings from different datasets as though they were synchronized sensors.

### C. Urban GNSS/inertial fusion

Candidate data: UrbanNav. Learn per-observation or solution-level reliability and bias, then assess positioning, velocity/attitude where supported, outages, recovery, and uncertainty. Start by choosing tightly coupled raw-GNSS fusion versus loosely coupled position-solution fusion: they are different tasks.

CNN/LSTM/transformer adaptation is already represented in this literature. Generalization between routes, receivers, and obstruction types, and comparison to robust adaptive classical methods, are more meaningful contributions than a network-family substitution.

### D. Spacecraft temporal pose filtering

Candidate data: SHIRT, with SPEED+ useful for visual training. Compare neural pose measurements plus an adaptive UKF against a direct temporal network and an uncertainty-aware differentiable model. Geometry, dynamics, visual ambiguity, and simulation-to-laboratory transfer offer substantive research questions.

This is a good application-driven choice, but the measurement suite and limited trajectory diversity make it a different undertaking from broadly sampled IMU/GNSS or UWB sensor fusion. Additional simulated sensors must be labeled as simulation.

### E. Vehicle dynamics or truck states

Sideslip, articulation, and tire-force inference are state-estimation problems directly relevant to control, even without reconstructing a global path. This is attractive if those states are more interesting than navigation pose. The main selection hurdle is a sufficiently rich, reusable dataset with credible reference instruments and changing operating conditions. KITTI pose labels do not provide tire friction, tire force, or sideslip truth by default.

## 9. Evaluation design that makes comparisons credible

1. **Declare state and measurements.** List what is estimated, what is observed, what is initialized from truth, and what can only be scored indirectly. Keep reference channels outside inference-time inputs.
2. **Separate causal inference from smoothing.** Centered windows, bidirectional networks, interpolation using future samples, and ground-truth-derived preprocessing can change latency or leak information. Offline reference smoothing is acceptable for evaluation if disclosed.
3. **Split by complete experiments.** Hold out trajectories, collection sessions, geometries, robots, or routes. Randomly splitting overlapping windows produces an overly easy test. Cross-validation still needs a distinct procedure for hyperparameter selection.
4. **Use matched information.** Give competing methods the same sensor streams, calibration, initialization, and latency. Do not compare a learned system with oracle calibration against a raw classical baseline.
5. **Use strong relevant baselines.** A tuned ESKF/invariant EKF or UKF, robust residual gating/losses, and a classical adaptive covariance method are sensible starting points. Add SFUISE for UWB timing/calibration questions, graph optimization for navigation, and IMM for switching dynamics where appropriate. UKF is not inherently superior to a well-designed ESKF.
6. **Compare direct and structured learning.** Include a causal sequence network, separately trained learned measurements/covariances, and end-to-end trained hybrids. Ablate bias correction, covariance adaptation, dynamics residuals, and physics losses individually.
7. **Score the state, not just the plotted path.** Position RMSE/ATE, velocity error where reference supports it, orientation geodesic error, bias error only against a clearly qualified reference, and divergence/outage recovery all matter. Relative trajectory error is useful but insufficient on its own.
8. **State alignment rules.** Do not remove all absolute localization error by fitting the whole test trajectory to truth. Use calibration or an explicit initial-frame convention for externally anchored tasks. Gauge alignment remains necessary and legitimate for appropriate odometry comparisons.
9. **Assess uncertainty.** Use likelihood, empirical coverage, NIS, and NEES where corresponding truth is defensible. Rotational errors belong in a tangent space. Exact chi-square claims require their distributional assumptions; adjacent time samples are not independent Monte Carlo trials.
10. **Test failure and deployment cost.** Missing observations, realistic biases, delays, changed geometry, and out-of-distribution motion should accompany mean error. Report tail errors, failed runs, compute, memory, and actual inference latency.

Learning cannot guarantee bounded position drift through indefinitely missing external information. A useful estimator should expose increasing uncertainty and recover sensibly when information returns.

## 10. Recommended discussion sequence

First decide whether the central physical question is **measurement reliability**, **dynamics/model mismatch**, or **latent physical-state inference from raw modalities**. That choice should guide the dataset.

My provisional recommendation is to inspect MILUV and UTIL most closely for a 3D navigation project, with UrbanNav as the strongest road-navigation alternative. Preserve NeuroBEM and SHIRT as serious branches if the motivating interest is aerodynamic physics or spacecraft, respectively.

A productive first paper set is R01, R04, R06, R14, R15, R28, and the selected dataset paper. Then use R02/R17 for residual dynamics, R16 for urban GNSS, or R20/R21 for spacecraft. No novelty claim follows from this scoping review; a narrower prior-art search is needed after selecting a concrete research question.

## 11. Official dataset/source index

- **D01 — MILUV:** [official documentation](https://www.decar.ca/miluv/), [paper](https://arxiv.org/abs/2504.14376), [2026 journal publication](https://doi.org/10.1177/02783649251405898), [devkit](https://github.com/decargroup/miluv).
- **D02 — UTIL:** [official documentation](https://learnsyslab.github.io/util-uwb-dataset/), [paper](https://arxiv.org/abs/2203.14471), [2024 journal publication](https://doi.org/10.1177/02783649241230640), [baseline driver](https://raw.githubusercontent.com/utiasDSL/util-uwb-dataset/main/scripts/estimation/main.py), [ESKF implementation](https://raw.githubusercontent.com/utiasDSL/util-uwb-dataset/main/scripts/estimation/eskf_class.py).
- **D03 — UrbanNav:** [official repository](https://github.com/IPNL-POLYU/UrbanNavDataset), [laboratory overview](https://www.polyu.edu.hk/aae/ipn-lab/us/en/projects/urbannav/), [2021 conference paper](https://doi.org/10.33012/2021.17895), [2023 journal paper](https://doi.org/10.33012/navi.602).
- **D04 — NTU VIRAL:** [official dataset](https://ntu-aris.github.io/ntu_viral_dataset/), [sensor/reference documentation](https://ntu-aris.github.io/ntu_viral_dataset/sensors_and_usage.html).
- **D05 — EuRoC MAV:** [official dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/), [OpenVINS dataset/initialization/reference guide](https://docs.openvins.com/gs-datasets.html), [OpenVINS installation guidance](https://docs.openvins.com/gs-installing.html).
- **D06 — TUM VI:** [official dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset).
- **D07 — Boreas:** [official data reference](https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md), [devkit/download/evaluation entry point](https://github.com/utiasASRL/pyboreas).
- **D08 — NCLT:** [official dataset and reference-provenance notes](https://robots.engin.umich.edu/nclt/index.html).
- **D09 — M2DGR:** [official project](https://sjtu-visys.github.io/M2DGR/), [repository with evaluated systems and dataset guidance](https://github.com/SJTU-ViSYS/M2DGR), [benchmark framework linked by the authors](https://github.com/SJTU-ViSYS/M2DGR-Benchmark).
- **D10 — NeuroBEM:** [official project](https://rpg.ifi.uzh.ch/NeuroBEM.html), [data documentation](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html).
- **D11 — Blackbird:** [dataset paper](https://doi.org/10.1177/0278364920908331), [official repository and current availability note](https://github.com/mit-aera/Blackbird-Dataset), [ISER proceedings paper](https://doi.org/10.1007/978-3-030-33950-0_12).
- **D12 — SPEED+:** [Stanford dataset overview](https://slab.stanford.edu/projects/datasets), [repository record](https://purl.stanford.edu/wv398fc4383), [2022 IEEE Aerospace dataset paper](https://doi.org/10.1109/AERO53065.2022.9843439).
- **D13 — SHIRT:** [author project and paper](https://taehajeffpark.com/shirt/).
- **D14 — Newer College:** [official dataset](https://ori-drs.github.io/newer-college-dataset/), [reference construction](https://ori-drs.github.io/newer-college-dataset/ground-truth/).
- **D15 — AQUALOC:** [official dataset](https://www.lirmm.fr/aqualoc/).
- **D16 — KITTI (common comparison, not first recommendation):** [raw-data documentation](https://www.cvlibs.net/datasets/kitti/raw_data.php), [official odometry metrics/devkit/submission policy](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [2013 data paper](https://doi.org/10.1177/0278364913491297). Stereo, LiDAR, and OXTS GPS/IMU products support extensive navigation literature. Raw and odometry benchmark products differ; pose-derived inputs require care. The dataset is not a default source of independently labeled physical dynamics states.
